# Creating a basic attention mechanism

import torch
import torch.nn as nn
from einops import einsum, rearrange

torch.manual_seed(42)

inputs = torch.tensor(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
    dtype=torch.float32,
)

batch = torch.stack((inputs, inputs), dim=0)
# print(batch.shape)

x_2 = inputs[1]

d_in = 3
d_out = 2

torch.manual_seed(123)

w_key = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
w_query = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
w_value = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)

keys = inputs @ w_key
queries = inputs @ w_query
values = inputs @ w_value

attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
# #print(attn_weights)
context_vec = attn_weights @ values

# #print(context_vec)
# #print("\nClass version:\n")


# turning into a class
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_key = nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
        self.W_query = nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
        self.W_value = nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T  # ω = omega = raw attention scores
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )  # α = alpha = softmax to get attention weights
        context_vec = attn_weights @ values
        return context_vec


# sa_v1 = SelfAttention_v1(d_in, d_out)
# #print(sa_v1(inputs))


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T  # ω = omega = raw attention scores
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )  # α = alpha = softmax to get attention weights
        context_vec = attn_weights @ values
        return context_vec


sa_v2 = SelfAttention_v2(d_in, d_out, qkv_bias=False)
# print(sa_v2(inputs))

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
values = sa_v2.W_value(inputs)

attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
# print(attn_weights)

context_length = attn_weights.shape[1]
mask_efficient = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_weights.masked_fill(mask_efficient.bool(), -torch.inf)  # type: ignore[arg-type]
# print(masked)

normalized = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
# print(normalized)

context_vec = normalized @ values
# print(context_vec)

# dropouts
dropout = torch.nn.Dropout(0.5)
# print(dropout(attn_weights))


# Causal Attention class
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(-2, -1)

        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # type: ignore[arg-type]

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec


# Causal Attention class (using einsum)
class CausalAttentionEinsum(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch, num_tokens, d_in = x.shape
        keys = self.W_key(x)  # (batch, seq, dim)
        queries = self.W_query(x)  # (batch, seq, dim)
        values = self.W_value(x)  # (batch, seq, dim)

        # For each batch, compute dot product between query at position i and key at position j
        attn_scores = einsum(
            "batch query dim, batch key dim -> batch query key", queries, keys
        )

        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # type: ignore[arg-type]

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)

        # For each batch and query position, compute weighted sum of values across all key positions
        context_vec = einsum(
            "batch query key, batch key dim -> batch query dim", attn_weights, values
        )
        return context_vec


# context_length = batch.shape[1]
# ca = CausalAttention(d_in, d_out, context_length, 0.0, qkv_bias=False)
# context_vecs = ca(batch)
# print(context_vecs)
# print("context_vecs.shape:", context_vecs.shape)


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(
                    d_in, d_out, context_length, dropout, qkv_bias
                )  # THIS IS SEQUENTIAL MULTIHEADED ATTENTION; In the future we will parallelize this.
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


context_length = batch.shape[1]
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2, qkv_bias=False
)
# context_vecs = mha(batch)
# print(context_vecs)
# print("context_vecs.shape:", context_vecs.shape)


# Parallel MultiHeaded Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch, num_tokens, d_in = x.shape

        # Project to queries, keys, values
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Split into multiple heads: (batch, seq, d_out) -> (batch, seq, heads, head_dim)
        queries = rearrange(
            queries,
            "batch seq (heads dim) -> batch seq heads dim",
            heads=self.num_heads,
        )
        keys = rearrange(
            keys, "batch seq (heads dim) -> batch seq heads dim", heads=self.num_heads
        )
        values = rearrange(
            values, "batch seq (heads dim) -> batch seq heads dim", heads=self.num_heads
        )

        # Compute attention scores per head
        # (batch, seq, heads, dim) -> (batch, heads, query, key)
        attention_scores = einsum(
            queries,
            keys,
            "batch query heads dim, batch key heads dim -> batch heads query key",
        )

        # Apply causal mask
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],  # type: ignore[arg-type]
            -torch.inf,  # type: ignore[arg-type]
        )

        # Softmax and dropout
        attention_weights = torch.softmax(attention_scores / self.head_dim**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values per head
        # (batch, heads, query, key) @ (batch, key, heads, dim) -> (batch, query, heads, dim)
        context_vec = einsum(
            attention_weights,
            values,
            "batch heads query key, batch key heads dim -> batch query heads dim",
        )

        # Merge heads back: (batch, seq, heads, head_dim) -> (batch, seq, d_out)
        context_vec = rearrange(
            context_vec, "batch seq heads dim -> batch seq (heads dim)"
        )

        # Output projection
        context_vec = self.out_proj(context_vec)
        return context_vec


batch2 = torch.randn(2, 1024, 768)
mha = MultiHeadAttention(768, 768, 1024, 0.0, num_heads=12, qkv_bias=False)
context_vecs = mha(batch2)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
