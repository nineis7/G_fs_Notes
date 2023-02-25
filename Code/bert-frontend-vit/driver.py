import tvm.relay
from transformers import BertModel, BertTokenizer, BertConfig
import torch
import numpy as np
import os
import time

enc = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenizing input text
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = enc.tokenize(text)
# print("tokenized_text: ", tokenized_text)

# Masking one of the input tokens
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

segments = np.array([segments_ids]).astype(np.int32)
path = "dump"
if not os.path.exists(path):
    os.makedirs(path)
np.save(path + "/attention_mask.npy", attention_mask)
np.save(path + "/token_type_ids.npy", segments_ids)
np.save(path + "/input_id.npy", indexed_tokens)

# Creating a dummy input
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensor = torch.tensor([segments_ids])
attention_mask_tensor = torch.tensor([attention_mask])
dummy_input = [tokens_tensor, segments_tensor]


# real input from imdb dataset 
from datasets import load_dataset
raw_datasets = load_dataset("imdb")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#inputs = tokenizer(sentences, padding="max_length", truncation=True)
def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=64, padding="max_length", truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
from torch.utils.data import DataLoader
BATCH_SIZE = 1
eval_dataloader = DataLoader(small_eval_dataset, batch_size=BATCH_SIZE)
for batch in eval_dataloader:
    indexed_tokens = [_.numpy().tolist() for _ in batch['input_ids']]
    indexed_tokens = np.array(indexed_tokens).T.tolist()
    tokens_tensor = torch.tensor(indexed_tokens)
    np.save(os.path.join(path, 'input_id.npy'), indexed_tokens)
    attention_mask = [_.numpy().tolist() for _ in batch['attention_mask']]
    attention_mask = np.array(attention_mask).T.tolist()
    attention_mask_tensor = torch.tensor(attention_mask)
    np.save(os.path.join(path, 'attention_mask.npy'), attention_mask)
    segments_ids = [_.numpy().tolist() for _ in batch['token_type_ids']]
    segments_ids = np.array(segments_ids).T.tolist()
    segments_tensor = torch.tensor(segments_ids)
    np.save(os.path.join(path, 'token_type_ids.npy'), segments_ids)
    break

# Initializing the model with the torchscript flag
# Flag set to True even though it is not necessary as this model does not have an LM Head.
config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
    num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, torchscript=True)

# Instantiating the model
model = BertModel(config)

# The model needs to be in evaluation mode
model.eval()
# print(model.__dict__)

# If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)
start = time.time()
output = model(tokens_tensor, attention_mask_tensor, token_type_ids=segments_tensor)
end = time.time()
print("inference time: ", end-start, "seconds")
# import ipdb; ipdb.set_trace()

# golden case for pytorch example
print("output[0]: ", output[0].shape)
print("output[1]: ", output[1].shape)
path = "golden_layer"
if not os.path.exists(path):
    os.makedirs(path)
np.save(path + "/output_1.npy", output[0].detach().numpy())
np.save(path + "/output_2.npy", output[1].detach().numpy())

# Creating the trace
traced_model = torch.jit.trace(model, [tokens_tensor, attention_mask_tensor, segments_tensor])
shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
# parse pytorch model to tvm relay ir
mod, params = tvm.relay.frontend.pytorch.from_pytorch(traced_model, shape_list, default_dtype="float32")
# print(mode) to check relay ir
from tvm.relay.build_module import BuildModule
opt_level = 3
target = "llvm"
with tvm.transform.PassContext(opt_level=opt_level):
    module = BuildModule()
    # optimize() is where we will do operator fusion and quatization
    module.optimize_custom(mod, target=target, params=params)

'''
# somehow cannot export and reload, so we keep all above for now
loaded_model = torch.jit.load("traced_bert.pt")
loaded_model.eval()

all_encoder_layers, pooled_output = loaded_model(dummy_input)
print(all_encoder_layers)
print(pooled_output)
'''
