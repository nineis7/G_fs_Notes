"""
---
title: Relative Multi-Headed Attention
summary: >
  Documented implementation with explanations of
  Relative Multi-Headed Attention from paper Transformer-XL.
---

# Relative Multi-Headed Attention

This is an implementation of relative multi-headed attention from paper
[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://papers.labml.ai/paper/1901.02860)
in [PyTorch](https://pytorch.org).
"""

import torch
from torch import nn

from labml.logger import inspect
# from labml_nn.transformers.mha import MultiHeadAttention
from labml_helpers.module import Module
from typing import Optional, List

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
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        """
        `mask` has shape `[seq_len_q, seq_len_k, batch_size]`, where first dimension is the query dimension.
        If the query dimension is equal to $1$ it will be broadcasted.
        """
        print("mask.shape: ", mask.shape)
        print("query_shape: ", query_shape)
        print("key_shape: ", key_shape)
        
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
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        # Save attentions for any other calculations 
        self.attn = attn.detach()

        # Concatenate multiple heads
        x = x.reshape(seq_len, batch_size, -1)

        # Output layer
        return self.output(x)

def shift_right(x: torch.Tensor):
    """
    This method shifts $i^{th}$ row of a matrix by $i$ columns.

    If the input is `[[1, 2 ,3], [4, 5 ,6], [7, 8, 9]]`, the shifted
    result would be `[[1, 2 ,3], [0, 4, 5], [9, 0, 7]]`.
    *Ideally we should mask out the lower triangle but it's ok for our purpose*.
    """

    # Concatenate a column of zeros
    zero_pad = x.new_zeros(x.shape[0], 1, *x.shape[2:])
    x_padded = torch.cat([x, zero_pad], dim=1)

    # Reshape and remove excess elements from the end
    x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])
    x = x_padded[:-1].view_as(x)

    #
    return x


class RelativeMultiHeadAttention(MultiHeadAttention):
    """
    ## Relative Multi-Head Attention Module

    We override [Multi-Head Attention](mha.html) module so we only need to 
    write the `get_scores` method.
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        # The linear transformations do not need a bias since we
        # explicitly include it when calculating scores.
        # However having a bias for `value` might make sense.
        super().__init__(heads, d_model, dropout_prob, bias=False)

        # Number of relative positions
        self.P = 2 ** 12

        # Relative positional embeddings for key relative to the query.
        # We need $2P$ embeddings because the keys can be before or after the query.
        self.key_pos_embeddings = nn.Parameter(torch.zeros((self.P * 2, heads, self.d_k)), requires_grad=True)
        # Relative positional embedding bias for key relative to the query.
        self.key_pos_bias = nn.Parameter(torch.zeros((self.P * 2, heads)), requires_grad=True)
        # Positional embeddings for the query is independent of the position of the query
        self.query_pos_bias = nn.Parameter(torch.zeros((heads, self.d_k)), requires_grad=True)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        r"""
        ### Get relative attention scores

        With absolute attention

        \begin{align}
        A^{abs}_{j} &= lin_q(X^q_i + P_i)^\top lin_k(X^k_j + P_j) \\
                      &= \underset{\textcolor{lightgreen}{A}}{Q_i^\top K_j} +
                         \underset{\textcolor{lightgreen}{B}}{Q_i^\top U^K_j} +
                         \underset{\textcolor{lightgreen}{C}}{{U^Q_i}^\top K_j} +
                         \underset{\textcolor{lightgreen}{D}}{{U^Q_i}^\top U^K_j}
        \end{align}

        where $Q_i, K_j$, are linear transformations of
         original embeddings $X^q_i, X^k_j$
         and $U^Q_i, U^K_j$ are linear transformations of
         absolute positional encodings $P_i, P_j$.

        They reason out that the attention to a given key should be the same regardless of
        the position of query.
        Hence replace $\underset{\textcolor{lightgreen}{C}}{{U^Q_i}^\top K_j}$
        with a constant $\underset{\textcolor{lightgreen}{C}}{\textcolor{orange}{v^\top} K_j}$.

        For the second and third terms relative positional encodings are introduced.
        So $\underset{\textcolor{lightgreen}{B}}{Q_i^\top U^K_j}$ is
        replaced with $\underset{\textcolor{lightgreen}{B}}{Q_i^\top \textcolor{orange}{R_{i - j}}}$
        and $\underset{\textcolor{lightgreen}{D}}{{U^Q_i}^\top U^K_j}$
        with $\underset{\textcolor{lightgreen}{D}}{\textcolor{orange}{S_{i-j}}}$.

        \begin{align}
        A^{rel}_{i,j} &= \underset{\mathbf{\textcolor{lightgreen}{A}}}{Q_i^\top K_j} +
                         \underset{\mathbf{\textcolor{lightgreen}{B}}}{Q_i^\top \textcolor{orange}{R_{i - j}}} +
                         \underset{\mathbf{\textcolor{lightgreen}{C}}}{\textcolor{orange}{v^\top} K_j} +
                         \underset{\mathbf{\textcolor{lightgreen}{D}}}{\textcolor{orange}{S_{i-j}}}
        \end{align}
        """

        # $\textcolor{orange}{R_k}$
        key_pos_emb = self.key_pos_embeddings[self.P - key.shape[0]:self.P + query.shape[0]]
        # $\textcolor{orange}{S_k}$
        key_pos_bias = self.key_pos_bias[self.P - key.shape[0]:self.P + query.shape[0]]
        # $\textcolor{orange}{v^\top}$
        query_pos_bias = self.query_pos_bias[None, None, :, :]

        # ${(\textcolor{lightgreen}{\mathbf{A + C}})}_{i,j} =
        # Q_i^\top K_j +
        # \textcolor{orange}{v^\top} K_jZ$
        ac = torch.einsum('ibhd,jbhd->ijbh', query + query_pos_bias, key)
        # $\textcolor{lightgreen}{\mathbf{B'}_{i,k}} = Q_i^\top \textcolor{orange}{R_k}$
        b = torch.einsum('ibhd,jhd->ijbh', query, key_pos_emb)
        # $\textcolor{lightgreen}{\mathbf{D'}_{i,k}} = \textcolor{orange}{S_k}$
        d = key_pos_bias[None, :, None, :]
        # Shift the rows of $\textcolor{lightgreen}{\mathbf{(B' + D')}_{i,k}}$
        # to get $$\textcolor{lightgreen}{\mathbf{(B + D)}_{i,j} = \mathbf{(B' + D')}_{i,i - j}}$$
        bd = shift_right(b + d)
        # Remove extra positions
        bd = bd[:, -key.shape[0]:]

        # Return the sum $$
        # \underset{\mathbf{\textcolor{lightgreen}{A}}}{Q_i^\top K_j} +
        # \underset{\mathbf{\textcolor{lightgreen}{B}}}{Q_i^\top \textcolor{orange}{R_{i - j}}} +
        # \underset{\mathbf{\textcolor{lightgreen}{C}}}{\textcolor{orange}{v^\top} K_j} +
        # \underset{\mathbf{\textcolor{lightgreen}{D}}}{\textcolor{orange}{S_{i-j}}}
        # $$
        return ac + bd


def _test_shift_right():
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    inspect(x)
    inspect(shift_right(x))

    x = torch.arange(1, 6)[None, :, None, None].repeat(5, 1, 1, 1)
    inspect(x[:, :, 0, 0])
    inspect(shift_right(x)[:, :, 0, 0])

    x = torch.arange(1, 6)[None, :, None, None].repeat(3, 1, 1, 1)
    inspect(x[:, :, 0, 0])
    inspect(shift_right(x)[:, :, 0, 0])


if __name__ == '__main__':
    _test_shift_right()
