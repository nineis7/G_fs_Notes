"""
---
title: Switch Transformer Experiment
summary: This experiment trains a small switch transformer on tiny Shakespeare dataset.
---

# Switch Transformer Experiment

This is an annotated PyTorch experiment to train a switch transformer.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/switch/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/353770ce177c11ecaa5fb74452424f46)
"""

import torch
import torch.nn as nn
import math
import copy
import numpy as np

from typing import Callable

from labml import experiment, tracker
from labml.configs import option
from labml_helpers.module import Module
from labml_helpers.train_valid import BatchIndex
# from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
from labml_helpers.datasets.text import TextDataset, SequentialDataLoader, SequentialUnBatchedDataset, TextFileDataset
from labml_helpers.device import DeviceConfigs
from labml_helpers.train_valid import TrainValidConfigs, hook_model_outputs, BatchIndex
from labml_helpers.metrics.accuracy import Accuracy
from torch.utils.data import DataLoader, RandomSampler

from labml.logger import Text
from labml import monit
# from .experiment import Configs
# from .experiment import switch_transformer

from labml_helpers.module import Module

from labml_helpers.module import M, TypedModuleList
from typing import List, Dict, Optional

def text_to_i(text: str):
    """
    Transform the text into a tensor of ids
    """
    return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

def clone_module_list(module: M, n: int) -> TypedModuleList[M]:
    """
    ## Clone Module

    Make a `nn.ModuleList` with clones of a given module
    """
    return TypedModuleList([copy.deepcopy(module) for _ in range(n)])

class PrepareForMultiHeadAttention(Module):
    """
    <a id="PrepareMHA"></a>

    ## Prepare for multi-head attention

    This module does a linear transformation and splits the vector into given
    number of heads for multi-head attention.
    This is used to transform **key**, **query**, and **value** vectors.
    """

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        # Linear layer for linear transform
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        # Number of heads
        self.heads = heads
        # Number of dimensions in vectors in each head
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # Input has shape `[seq_len, batch_size, d_model]` or `[batch_size, d_model]`.
        # We apply the linear transformation to the last dimension and split that into
        # the heads.
        head_shape = x.shape[:-1]

        # Linear transform
        x = self.linear(x)

        # Split last dimension into heads
        x = x.view(*head_shape, self.heads, self.d_k)

        # Output has shape `[seq_len, batch_size, heads, d_k]` or `[batch_size, d_model]`
        return x


class MultiHeadAttention(Module):
    r"""
    <a id="MHA"></a>

    ## Multi-Head Attention Module

    This computes scaled multi-headed attention for given `query`, `key` and `value` vectors.

    $$\mathop{Attention}(Q, K, V) = \underset{seq}{\mathop{softmax}}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$

    In simple terms, it finds keys that matches the query, and gets the values of
     those keys.

    It uses dot-product of query and key as the indicator of how matching they are.
    Before taking the $softmax$ the dot-products are scaled by $\frac{1}{\sqrt{d_k}}$.
    This is done to avoid large dot-product values causing softmax to
    give very small gradients when $d_k$ is large.

    Softmax is calculated along the axis of of the sequence (or time).
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        """
        * `heads` is the number of heads.
        * `d_model` is the number of features in the `query`, `key` and `value` vectors.
        """

        super().__init__()

        # Number of features per head
        self.d_k = d_model // heads
        # Number of heads
        self.heads = heads

        # These transform the `query`, `key` and `value` vectors for multi-headed attention.
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        # Softmax for attention along the time dimension of `key`
        self.softmax = nn.Softmax(dim=1)

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        # Scaling factor before the softmax
        self.scale = 1 / math.sqrt(self.d_k)

        # We store attentions so that it can be used for logging, or other computations if needed
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### Calculate scores between queries and keys

        This method can be overridden for other variations like relative attention.
        """

        # Calculate $Q K^\top$ or $S_{ijbh} = \sum_d Q_{ibhd} K_{jbhd}$
        # numpy
        # query = np.transpose(query.detach().numpy(), (1, 2, 0, 3))
        # key = np.transpose(key.detach().numpy(), (1, 2, 3, 0))
        # score = torch.tensor(np.transpose((query @ key), (2, 3, 0, 1)))
        query = query.permute(1, 2, 0, 3)
        key = key.permute(1, 2, 3, 0)
        score = (query @ key).permute(2, 3, 0, 1)
        # return torch.einsum('ibhd,jbhd->ijbh', query, key)
        return score

    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        """
        `mask` has shape `[seq_len_q, seq_len_k, batch_size]`, where first dimension is the query dimension.
        If the query dimension is equal to $1$ it will be broadcasted.
        """
        # print("mask.shape: ", mask.shape)
        # print("query_shape: ", query_shape)
        # print("key_shape: ", key_shape)
        
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        # Same mask applied to all heads.
        mask = mask.unsqueeze(-1)

        # resulting mask has shape `[seq_len_q, seq_len_k, batch_size, heads]`
        return mask

    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        `query`, `key` and `value` are the tensors that store
        collection of *query*, *key* and *value* vectors.
        They have shape `[seq_len, batch_size, d_model]`.

        `mask` has shape `[seq_len, seq_len, batch_size]` and
        `mask[i, j, b]` indicates whether for batch `b`,
        query at position `i` has access to key-value at position `j`.
        """

        # `query`, `key` and `value`  have shape `[seq_len, batch_size, d_model]`
        seq_len, batch_size, _ = query.shape
        
        # print("mask here: ", mask)
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        # Prepare `query`, `key` and `value` for attention computation.
        # These will then have shape `[seq_len, batch_size, heads, d_k]`.
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Compute attention scores $Q K^\top$.
        # This gives a tensor of shape `[seq_len, seq_len, batch_size, heads]`.
        scores = self.get_scores(query, key)

        # Scale scores $\frac{Q K^\top}{\sqrt{d_k}}$
        scores *= self.scale

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # $softmax$ attention along the key sequence dimension
        # $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = self.softmax(scores)

        # Save attentions if debugging
        tracker.debug('attn', attn)

        # Apply dropout
        attn = self.dropout(attn)

        # Multiply by values
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$

        # x = torch.einsum("ijbh,jbhd->ibhd", attn, value)
        # print("x.shape: ", x.shape)

        # bhij @ bhjd -> bhid -> ibhd (numpy)
        # attn = np.transpose(attn.detach().numpy(), (2, 3, 0, 1))
        # value = np.transpose(value.detach().numpy(), (1, 2, 0, 3))
        # x = torch.tensor(np.transpose((attn @ value), (2, 0, 1, 3)))
        attn = attn.permute(2, 3, 0, 1)
        value = value.permute(1, 2, 0, 3)
        x = (attn @ value).permute(2, 0, 1, 3)

        # Save attentions for any other calculations 
        self.attn = attn.detach()

        # Concatenate multiple heads
        x = x.reshape(seq_len, batch_size, -1)

        # Output layer
        return self.output(x)

class FeedForward(Module):
    """
    ## FFN module
    """

    def __init__(self, d_model: int, d_ff: int,
                 dropout: float = 0.1,
                 activation=nn.ReLU(),
                 is_gated: bool = False,
                 bias1: bool = True,
                 bias2: bool = True,
                 bias_gate: bool = True):
        """
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability for the hidden layer
        * `is_gated` specifies whether the hidden layer is gated
        * `bias1` specified whether the first fully connected layer should have a learnable bias
        * `bias2` specified whether the second fully connected layer should have a learnable bias
        * `bias_gate` specified whether the fully connected layer for the gate should have a learnable bias
        """
        super().__init__()
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        # Hidden layer dropout
        self.dropout = nn.Dropout(dropout)
        # Activation function $f$
        self.activation = activation
        # Whether there is a gate
        self.is_gated = is_gated
        if is_gated:
            # If there is a gate the linear layer to transform inputs to
            # be multiplied by the gate, parameterized by weight $V$ and bias $c$
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor):
        # $f(x W_1 + b_1)$
        g = self.activation(self.layer1(x))
        # If gated, $f(x W_1 + b_1) \otimes (x V + b) $
        if self.is_gated:
            x = g * self.linear_v(x)
        # Otherwise
        else:
            x = g
        # Apply dropout
        x = self.dropout(x)
        # $(f(x W_1 + b_1) \otimes (x V + b)) W_2 + b_2$ or $f(x W_1 + b_1) W_2 + b_2$
        # depending on whether it is gated
        return self.layer2(x)

class SwitchFeedForward(Module):
    """
    ## Routing among multiple FFNs
    """

    def __init__(self, *,
                 capacity_factor: float,
                 drop_tokens: bool,
                 is_scale_prob: bool,
                 n_experts: int,
                 expert: FeedForward,
                 d_model: int):
        """
        * `capacity_factor` is the capacity of each expert as a factor relative to ideally balanced load
        * `drop_tokens` specifies whether to drop tokens if more tokens are routed to an expert than the capacity
        * `is_scale_prob` specifies whether to multiply the input to the FFN by the routing probability
        * `n_experts` is the number of experts
        * `expert` is the expert layer, a [FFN module](../feed_forward.html)
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability in the FFN
        """
        super().__init__()

        self.capacity_factor = capacity_factor
        self.is_scale_prob = is_scale_prob
        self.n_experts = n_experts
        self.drop_tokens = drop_tokens

        # make copies of the FFNs
        self.experts = clone_module_list(expert, n_experts)
        # Routing layer and softmax
        self.switch = nn.Linear(d_model, n_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input to the switching module with shape `[seq_len, batch_size, d_model]`
        """

        # Capture the shape to change shapes later
        seq_len, batch_size, d_model = x.shape
        # Flatten the sequence and batch dimensions
        x = x.view(-1, d_model)

        # Get routing probabilities for each of the tokens.
        # $$p_i(x) = \frac{e^{h(x)_i}}{\sum^N_j e^{h(x)_j}}$$
        # where $N$ is the number of experts `n_experts` and
        # $h(\cdot)$ is the linear transformation of token embeddings.
        # print("self.switch(x): ", self.switch(x))
        route_prob = self.softmax(self.switch(x))

        # Get the maximum routing probabilities and the routes.
        # We route to the expert with highest probability
        # print("route_prob: ", route_prob)
        route_prob_max, routes = torch.max(route_prob, dim=-1)

        # Get indexes of tokens going to each expert
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]
        # indexes_list[2] = torch.tensor([2], dtype=torch.int64)
        print("indexes_list: ", indexes_list)

        # Initialize an empty tensor to store outputs
        # print("x.shape: ", x.shape)
        # final_output = x.new_zeros(x.shape)
        final_output = torch.zeros(x.shape)
        # print("output: ", final_output)
        # final_output = torch.tensor([[0] * 128 for i in range(5)], dtype=torch.float32)
        # print("final_output.shape: ", final_output.shape)
        # final_output = torch.unsqueeze(final_output, 0)
        # print("x2.shape: ", x.shape)

        # Capacity of each expert.
        # $$\mathrm{expert\;capacity} =
        # \frac{\mathrm{tokens\;per\;batch}}{\mathrm{number\;of\;experts}}
        # \times \mathrm{capacity\;factor}$$
        capacity = int(self.capacity_factor * len(x) / self.n_experts)
        # Number of tokens routed to each expert.
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])
        # print("counts: ", counts)

        # Initialize an empty list of dropped tokens
        dropped = []
        # Only drop tokens if `drop_tokens` is `True`.
        if self.drop_tokens:
            # Drop tokens in each of the experts
            for i in range(self.n_experts):
                # Ignore if the expert is not over capacity
                if len(indexes_list[i]) <= capacity:
                    continue
                # Shuffle indexes before dropping
                # print("len(indexes_list[i]): ", len(indexes_list[i]))
                # if len(indexes_list[i]) == 2:
                tmp = torch.tensor([1, 0], dtype=torch.long)
                # elif len(indexes_list[i]) == 3:
                #     tmp = torch.tensor([1, 0, 2], dtype=torch.float)
                # indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                # print(torch.randperm(len(indexes_list[i])))
                indexes_list[i] = indexes_list[i][tmp]
                # Collect the tokens over capacity as dropped tokens
                print("indexes_list[i][capacity:]: ", indexes_list[i][capacity:])
                dropped.append(indexes_list[i][capacity:])
                # Keep only the tokens upto the capacity of the expert
                indexes_list[i] = indexes_list[i][:capacity]

        # Get outputs of the expert FFNs
        expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]

        # Assign to final output
        for i in range(self.n_experts):
            final_output[indexes_list[i], :] = expert_output[i]

        # Pass through the dropped tokens
        if dropped:
            # print("in dropped")
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

        if self.is_scale_prob:
            # Multiply by the expert outputs by the probabilities $y = p_i(x) E_i(x)$
            final_output = final_output * route_prob_max.view(-1, 1)
        else:
            # Don't scale the values but multiply by $\frac{p}{\hat{p}} = 1$ so that the gradients flow
            # (this is something we experimented with).
            final_output = final_output * (route_prob_max / route_prob_max.detach()).view(-1, 1)

        # Change the shape of the final output back to `[seq_len, batch_size, d_model]`
        # print("seq_len, batch_size, d_model:", seq_len, batch_size, d_model)
        final_output = final_output.view(seq_len, batch_size, d_model)

        # Return
        #
        # * the final output
        # * number of tokens routed to each expert
        # * sum of probabilities for each expert
        # * number of tokens dropped.
        # * routing probabilities of the selected experts
        #
        # These are used for the load balancing loss and logging
        # print("final_output: ", final_output)
        print("dropped: ", dropped)
        return final_output, counts, route_prob.sum(0), len(dropped), route_prob_max


class SwitchTransformerLayer(Module):
    """
    # Switch Transformer Block

    This is the same as [normal transformer block](../models.html#TransformerLayer)
    with handling extra outputs of switch feedforward module.
    """

    def __init__(self, *,
                 d_model: int,
                 attn: MultiHeadAttention,
                 feed_forward: SwitchFeedForward,
                 dropout_prob: float):
        """
        * `d_model` is the token embedding size
        * `attn` is the attention module
        * `feed_forward` is the feed forward module (which is the switching module in this case)
        * `dropout_prob` is the probability of dropping out after self attention and FFN
        """
        super().__init__()
        self.size = d_model
        self.attn = attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

    def forward(self, *,
                x: torch.Tensor,
                mask: torch.Tensor):
        # print("x.shape: ", x.shape)
        # print("mask shape: ", mask.shape)
        # Normalize the vectors before doing self attention
        z = self.norm_self_attn(x)
        # Run through self attention, i.e. keys and values are from self
        self_attn = self.attn(query=z, key=z, value=z, mask=mask)
        # Add the self attention results
        x = x + self.dropout(self_attn)

        # Normalize for feed-forward
        z = self.norm_ff(x)
        # Pass through the switching feed-forward network
        ff, counts, route_prob, n_dropped, route_prob_max = self.feed_forward(z)
        # print("ff, counts, route_prob, n_dropped, route_prob_max: ", ff, counts, route_prob, n_dropped, route_prob_max)
        # Add the feed-forward results back
        x = x + self.dropout(ff)

        return x, counts, route_prob, n_dropped, route_prob_max


class SwitchTransformer(Module):
    """
    ## Switch Transformer
    """

    def __init__(self, layer: SwitchTransformerLayer, n_layers: int):
        super().__init__()
        # Make copies of the transformer layer
        self.layers = clone_module_list(layer, n_layers)
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # Run through each transformer layer
        counts, route_prob, n_dropped, route_prob_max = [], [], [], []
        for layer in self.layers:
            # print("mask: ", mask)
            x, f, p, n_d, p_max = layer(x=x, mask=mask)
            counts.append(f)
            route_prob.append(p)
            n_dropped.append(n_d)
            route_prob_max.append(p_max)
        # Finally, normalize the vectors
        x = self.norm(x)
        #
        return x

class CrossEntropyLoss(Module):
    """
    ### Cross entropy loss
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.loss(outputs.view(-1, outputs.shape[-1]), targets.view(-1))

class NLPAutoRegressionConfigs(TrainValidConfigs):
    """
    <a id="NLPAutoRegressionConfigs"></a>

    ## Trainer configurations

    This has the basic configurations for NLP auto-regressive task training.
    All the properties are configurable.
    """

    # Optimizer
    optimizer: torch.optim.Adam
    # Training device
    device: torch.device = DeviceConfigs()

    # Autoregressive model
    model: Module
    # Text dataset
    text: TextDataset
    # Batch size
    batch_size: int = 16
    # Length of the sequence, or context size
    seq_len: int = 512
    # Number of token in vocabulary
    n_tokens: int = 128
    # Tokenizer
    tokenizer: Callable = 'character'

    # Text prompt to start sampling (for illustration)
    prompt: str
    # The token separator when sampling (blank for character level tokenization)
    prompt_separator: str

    # Whether to periodically save models
    is_save_models = True

    # Loss function
    loss_func = CrossEntropyLoss()
    # Accuracy function
    accuracy = Accuracy()
    # Model embedding size
    d_model: int = 512
    # Gradient clipping
    grad_norm_clip: float = 1.0

    # Training data loader
    train_loader: DataLoader = 'shuffled_train_loader'
    # Validation data loader
    valid_loader: DataLoader = 'shuffled_valid_loader'

    # Data loaders shuffle with replacement
    dataloader_shuffle_with_replacement: bool = False

    def init(self):
        """
        ### Initialization
        """
        # Set tracker configurations
        tracker.set_scalar("accuracy.*", True)
        tracker.set_scalar("loss.*", True)
        # Add a hook to log module outputs
        hook_model_outputs(self.mode, self.model, 'model')
        # Add accuracy as a state module.
        # The name is probably confusing, since it's meant to store
        # states between training and validation for RNNs.
        # This will keep the accuracy metric stats separate for training and validation.
        self.state_modules = [self.accuracy]

    def other_metrics(self, output: torch.Tensor, target: torch.Tensor):
        """Override to calculate and log other metrics"""
        pass

    def step(self, batch: any, batch_idx: BatchIndex):
        """
        ### Training or validation step
        """

        # Set training/eval mode
        self.model.train(self.mode.is_train)

        # Move data to the device
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        # Update global step (number of tokens processed) when in training mode
        if self.mode.is_train:
            tracker.add_global_step(data.shape[0] * data.shape[1])

        # Whether to capture model outputs
        with self.mode.update(is_log_activations=batch_idx.is_last):
            # Get model outputs.
            # It's returning a tuple for states when using RNNs.
            # This is not implemented yet. 😜
            output, *_ = self.model(data)

        # Calculate and log loss
        loss = self.loss_func(output, target)
        tracker.add("loss.", loss)

        # Calculate and log accuracy
        self.accuracy(output, target)
        self.accuracy.track()

        self.other_metrics(output, target)

        # Train the model
        if self.mode.is_train:
            # Calculate gradients
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)
            # Take optimizer step
            self.optimizer.step()
            # Log the model parameters and gradients on last batch of every epoch
            if batch_idx.is_last:
                tracker.add('model', self.model)
            # Clear the gradients
            self.optimizer.zero_grad()

        # Save the tracked metrics
        tracker.save()

    def sample(self):
        """
        ### Sampling function to generate samples periodically while training
        """

        # Starting prompt
        prompt = self.prompt
        # Collect output for printing
        log = [(prompt, Text.subtle)]
        # Sample 25 tokens
        for i in monit.iterate('Sample', 25):
            # Tokenize the prompt
            data = self.text.text_to_i(prompt).unsqueeze(-1)
            data = data.to(self.device)
            # Get the model output
            output, *_ = self.model(data)
            # Get the model prediction (greedy)
            output = output.argmax(dim=-1).squeeze()
            # Add the prediction to prompt
            prompt += self.prompt_separator + self.text.itos[output[-1]]
            # Add the prediction for logging
            log += [(self.prompt_separator + self.text.itos[output[-1]], Text.value)]

        # Print the sampled output
        logger.log(log)

class AutoregressiveModel(Module):
    """
    ## Auto regressive model
    """

    def __init__(self, n_vocab: int, d_model: int, transformer: Module):
        super().__init__()
        # Token embedding module
        self.src_embed = nn.Embedding(n_vocab, d_model)
        # Transformer
        self.transformer = transformer
        # Final layer
        self.generator = nn.Linear(d_model, n_vocab)
        self.mask = None

    def forward(self, x: torch.Tensor):
        # Initialize the subsequent mask
        print("self.mask: ", self.mask)
        if self.mask is None or self.mask.size(0) != len(x):
            from labml_nn.transformers.utils import subsequent_mask
            self.mask = subsequent_mask(len(x)).to(x.device)
        # Token embeddings
        x = self.src_embed(x)
        # Run it through the transformer
        res, counts, route_prob, n_dropped, route_prob_max = self.transformer(x, self.mask)
        # Generate logits of the next token
        res = self.generator(res)
        #
        return res, counts, route_prob, n_dropped, route_prob_max


class Configs(NLPAutoRegressionConfigs):
    """
    ## Configurations

    This extends [`NLPAutoRegressionConfigs`](../../experiments/nlp_autoregression.html).

    The default configs can and will be over-ridden when we start the experiment
    """

    # model: AutoregressiveModel
    transformer: Module

    # Token embedding size
    d_model: int = 128
    # Number of attention heads
    heads: int = 4
    # Dropout probability
    dropout: float = 0.0
    # Number of features in FFN hidden layer
    d_ff: int = 256
    # Number of transformer layers
    n_layers: int = 6
    # Number of experts
    n_experts: int = 4
    # Load balancing coefficient
    load_balancing_loss_ceof = 0.01
    # Whether to scale the chosen expert outputs by the routing probability
    is_scale_prob: bool = True
    # Whether to drop tokens
    drop_tokens: bool = True
    # Capacity factor to determine capacity of each model
    capacity_factor: float = 1.2

    def init(self):
        super().init()
        # Initialize tracking indicators
        tracker.set_scalar("lb_loss.*", False)
        tracker.set_scalar("route.*", False)
        tracker.set_scalar("dropped.*", False)

    def step(self, batch: any, batch_idx: BatchIndex):
        """
        ### Training or validation step
        """

        # Move data to the device
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        # Update global step (number of tokens processed) when in training mode
        if self.mode.is_train:
            tracker.add_global_step(data.shape[0] * data.shape[1])

        # Whether to capture model outputs
        with self.mode.update(is_log_activations=batch_idx.is_last):
            # Get model outputs.
            output, counts, route_prob, n_dropped, route_prob_max = self.model(data)

        # Calculate and cross entropy loss
        cross_entropy_loss = self.loss_func(output, target)
        # Total number of tokens processed, $T$, in the current batch $\mathscr{B}$
        total = counts.sum(dim=-1, keepdims=True)
        # Fraction of tokens routed to each expert
        # $$f_i = \frac{1}{T} \sum_{x \in \mathscr{B}} \mathbf{1} \{ \mathop{argmax} p(x), i \}$$
        # $f_i$ is the count of tokens where the argmax of $p(x)$ is equal to $i$.
        route_frac = counts / total
        # Mean routing probability
        # $$P_i = \frac{1}{T} \sum_{x \in \mathscr{B}} p_i (x)$$
        route_prob = route_prob / total
        # Load balancing loss
        # $$\mathscr{L} = N \sum_{i=1}^N f_i \cdot P_i$$
        # $\mathscr{L}$ is the loss for a single layer and here we are
        # taking the sum of losses across all layers.
        load_balancing_loss = self.n_experts * (route_frac * route_prob).sum()

        # Track stats
        tracker.add('dropped.', total.new_tensor(n_dropped) / total)
        tracker.add('route.min.', route_frac.min())
        tracker.add('route.max.', route_frac.max())
        tracker.add('route.std.', route_frac.std())
        tracker.add('route.max_prob.', route_prob_max)
        tracker.add("loss.", cross_entropy_loss)
        tracker.add("lb_loss.", load_balancing_loss)

        # Combined loss.
        # The load balancing loss is multiplied by a coefficient $\alpha$ which is
        # set to something small like $\alpha = 0.01$.
        loss = cross_entropy_loss + self.load_balancing_loss_ceof * load_balancing_loss

        # Calculate and log accuracy
        self.accuracy(output, target)
        self.accuracy.track()

        # Train the model
        if self.mode.is_train:
            # Calculate gradients
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)
            # Take optimizer step
            self.optimizer.step()
            # Log the model parameters and gradients on last batch of every epoch
            if batch_idx.is_last:
                tracker.add('model', self.model)
            # Clear the gradients
            self.optimizer.zero_grad()

        # Save the tracked metrics
        tracker.save()


@option(Configs.model)
def autoregressive_model(c: Configs):
    """
    ### Initialize the auto-regressive model
    """
    print("c.n_tokens: ", c.n_tokens)
    m = AutoregressiveModel(c.n_tokens, c.d_model, c.transformer)
    return m.to(c.device)


@option(Configs.transformer)
def switch_transformer(c: Configs):
    """
    ### Initialize the switch transformer
    """
    # from labml_nn.transformers.switch import SwitchTransformer, SwitchTransformerLayer, SwitchFeedForward
    # from labml_nn.transformers import MultiHeadAttention
    # from labml_nn.transformers.feed_forward import FeedForward
    print("configs: ", c.d_model, c.heads, c.d_model, c.dropout, c.capacity_factor, c.drop_tokens, c.is_scale_prob, c.n_experts, c.d_ff)
    return SwitchTransformer(
        SwitchTransformerLayer(d_model=c.d_model,
                               attn=MultiHeadAttention(c.heads, c.d_model, c.dropout),
                               feed_forward=SwitchFeedForward(capacity_factor=c.capacity_factor,
                                                              drop_tokens=c.drop_tokens,
                                                              is_scale_prob=c.is_scale_prob,
                                                              n_experts=c.n_experts,
                                                              expert=FeedForward(c.d_model, c.d_ff, c.dropout),
                                                              d_model=c.d_model),
                               dropout_prob=c.dropout),
        c.n_layers)


def main():
    """
    ### Run the experiment
    """
    # Create experiment
    experiment.create(name="switch_transformer", comment='')
    # Create configs
    conf = Configs()
    # Load configurations
    experiment.configs(conf,
                       # A dictionary of configurations to override
                       {'tokenizer': 'character',
                        'text': 'tiny_shakespeare',
                        'optimizer.learning_rate': 1.,
                        'optimizer.optimizer': 'Noam',
                        'prompt': 'It is',
                        'prompt_separator': '',

                        'transformer': 'switch_transformer',
                        'n_experts': 4,

                        'drop_tokens': True,
                        'capacity_factor': 1.2,

                        'train_loader': 'shuffled_train_loader',
                        'valid_loader': 'shuffled_valid_loader',

                        'seq_len': 64,
                        'epochs': 128,
                        'batch_size': 32,
                        'inner_iterations': 25,
                        })

    # Set models for saving and loading
    experiment.add_pytorch_models({'model': conf.model})

    # Start the experiment
    with experiment.start():
        # `TrainValidConfigs.run`
        conf.run()


#
if __name__ == '__main__':
    main()
