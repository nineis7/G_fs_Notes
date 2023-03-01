[TOC]



# Transformers Model Training & Result

这是blog的第二部分模型训练部分

## Training

some tools using for training

#### Batches and Masking

```python
class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        # 用特殊标识<blank>来表示padding，将其他有效部分的src扩展成attn_mask标准格式，即扩展成三维
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            # 除tgt第二维最后一行，最后一行为<end of sentence>
            self.tgt_y = tgt[:, 1:]
            # 除tgt第二维第一行，第一行为<start of sentence>
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            # 标准mask即上述倒三角矩阵
            self.ntokens = (self.tgt_y != pad).data.sum()
            # 记录输出tgt_y总共的字符个数

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        # 求交来将tgt_mask转变为倒三角表示以满足自回归特性
        return tgt_mask
```

如前所述，src_mask的维度为(batch, 1, time)，tgt_mask的维度为(batch, time, time)。

#### Training Loop

```python
class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed
```

```python
def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    # 对data_iter进行枚举遍历，i为当前batch的下标
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        # model.forward即EncoderDecoder class的forward
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state
```



