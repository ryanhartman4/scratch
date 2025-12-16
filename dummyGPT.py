import tiktoken
import torch
import torch.nn as nn
from einops import einsum, rearrange

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

GPT_CONFIG_345M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

GPT_CONFIG_774M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1280,
    "n_layers": 36,
    "n_heads": 20,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

GPT_CONFIG_1558M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1600,
    "n_heads": 25,
    "n_layers": 48,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"]
        )  # gets token ids and converts to embeddings
        self.pos_emb = nn.Embedding(
            cfg["context_length"], cfg["emb_dim"]
        )  # gets position ids and converts to embeddings
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[
                DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])
            ]  # placeholder for transformer blocks
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])  # placeholder for layer norm
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )  # placeholder for output head

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x


class LayerNorm(
    nn.Module
):  # Layer normalization is done instead of batch normalization because it is more flexible and stable. It also can be used in a parallelized way in ways that batch normalization cannot be.
    def __init__(self, emb_dim, epsilon=1e-5):
        super().__init__()
        self.eps = epsilon  # prevent division by zero
        self.scale = nn.Parameter(torch.ones(emb_dim))  # scale factor
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # shift factor

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return (
            self.scale * norm_x + self.shift
        )  # apply scale and shift to the normalized features


class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(
                        torch.tensor(2.0 / torch.pi)
                    )  # the square root of 2/pi is a constant that is used to normalize the input
                    * (
                        x + 0.044715 * torch.pow(x, 3)
                    )  # the 0.044715 is a constant that is used to smooth the input
                )  # the tanh function is a hyperbolic tangent function that is used to smooth the input
            )
        )


class FeedForward(
    nn.Module
):  # FeedForward is a module that is used to apply a feedforward network to the input
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GeLU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_in = cfg["emb_dim"]
        self.d_out = cfg["emb_dim"]
        self.context_length = cfg["context_length"]
        self.num_heads = cfg["n_heads"]
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.qkv_bias = cfg["qkv_bias"]
        self.out_proj = nn.Linear(self.d_out, self.d_out)
        self.w_q = nn.Linear(self.d_in, self.d_out, bias=self.qkv_bias)
        self.w_k = nn.Linear(self.d_in, self.d_out, bias=self.qkv_bias)
        self.w_v = nn.Linear(self.d_in, self.d_out, bias=self.qkv_bias)
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(self.context_length, self.context_length), diagonal=1
            ),
        )
        self.head_dim = self.d_out // self.num_heads

    def forward(self, x):
        batch, num_tokens, d_in = x.shape

        # calculates Q, K, V
        queries, keys, values = self.w_q(x), self.w_k(x), self.w_v(x)

        # Split into multiple heads
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

        # matrix multiply
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


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_block = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

model = GPTModel(GPT_CONFIG_124M)
# logits = model(batch)
# print("Output shape:", logits.shape)
# print(logits)

# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params:,}")


# total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
# print(f"Total number of parameters considering weight tying: {total_params_gpt2:,}")

# total_size_bytes = total_params * 4
# total_size_mb = total_size_bytes / (1024 * 1024)
# print(f"Total size of the model: {total_size_mb:.2f} MB")


def generate_text(model, prompt, max_new_tokens, context_size):
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(prompt)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    model.eval()
    for _ in range(max_new_tokens):
        encoded_tensor = encoded_tensor[:, -context_size:]
        with torch.no_grad():
            logits = model(encoded_tensor)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        encoded_tensor = torch.cat([encoded_tensor, next_token], dim=1)
    return tokenizer.decode(encoded_tensor.squeeze(0).tolist())


# print(generate_text(model, "Hello, I am", 10, GPT_CONFIG_124M["context_length"]))
