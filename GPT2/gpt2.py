# %%
from turtle import forward
import tiktoken
import torch 
import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "ctx_len": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

# TODO: DataLoader

# %%
class SelfAttention_v1(nn.Module):
    # v1 uses `nn.Parameter`
    def __init__(self, d_in, d_out):
        super().__init__()
        self.WQ = nn.Parameter(torch.rand(d_in, d_out))
        self.WK = nn.Parameter(torch.rand(d_in, d_out))
        self.WV = nn.Parameter(torch.rand(d_in, d_out))
    def forward(self, x):
        queries = x @ self.WQ
        values = x @ self.WV
        keys = x @ self.WK

        attn_score = queries @ keys.T
        attn_wt = torch.softmax(
            attn_score / keys.shape[-1]**(0.5), dim=-1
        )

        context_vct = attn_score @ values
        return context_vct

class SelfAttention_v2(nn.Module):
    # v2 uses `nn.Linear` which has automatic para init, optional bias terms
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.WQ = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.WK = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.WV = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.WQ(x)
        values = self.WV(x)
        keys = self.WK(x)

        attn_score = queries @ keys.T
        attn_wt = torch.softmax(
            attn_score / keys.shape[-1]**(0.5), dim=-1
        )

        context_vct = attn_score @ values
        return context_vct

class MHA(nn.Module):
    '''
    Note 1:
    Here, i can't reuse `SelfAttention` written above, as it written already collapsed single-head ctx_vec
    For MHA, 
        * per-head q/k/v projections should be reshaped as [B, H, L, HD]
        * then on head causal mask and softmax are applied separately
    ∀ B (batch size), head (no. of heads), L (sequence length), HD (head dimension)
    
    -----

    Note 2: Why is `register_buffer` used for mask instead of `self.mask`
    * PyTorch automatically modes the mask to correct device 
    * buffers are included in `state_dict` which are imp for training scenarios like saving and loading checkpoints
    * by-default grads aren't computed for buffer unlike regualar tensors
    
    -----

    Note 3: Diff `.T` and `.tanspose()`
    * .T(dim) transposes specific dim with last_dim (dim=-1)
    '''
    def __init__(
        self, d_in : int, d_out : int, 
        ctx_len : int, dropout, 
        num_heads : int, qkv_bias : bool =False
        ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.WQ = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.WK = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.WV = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)         # Layer combine head output
        self.dropout = nn.Dropout(dropout)
        
        # register_buffer is used to to register a fixed and non-trainable tensor
        # NOTE: Refer to "Note 2"
        self.register_buffer(
            name='mask',
            tensor=torch.triu(torch.ones(ctx_len, ctx_len), diagonal=1)
        )

    def forward(self, x):
        batch, num_tokens, d_in = x.shape

        # Shape: [B, L, E] i.e. [batch, num_tokens, d_in]
        queries = self.WQ(x)
        values = self.WV(x)
        keys = self.WK(x)

        # [batch, num_tokens, d_in] --> [batch, num_tokens, num_heads, head_dim] --> [batch, num_heads, num_tokens, head_dim]
        queries = queries.view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # NOTE: refer to `Note 3`
        attn_score = queries @ keys.transpose(2, 3)         # [num_tokens, head_dim] x [head_dim, num_tokens]
        
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_score.masked_fill_(mask_bool, -torch.inf)

        attn_wt = torch.softmax(attn_score/keys.shape[-1]**0.5, dim=-1)         # [batch, num_heads, num_tokens, num_tokens]     
        attn_wt = self.dropout(attn_wt)

        ctx_vec = (attn_wt @ values).transpose(1, 2)                            
        ctx_vec = ctx_vec.contiguous().view(batch, num_tokens, self.d_out)          # combining heads
        ctx_vec = self.out_proj(ctx_vec)

        return ctx_vec

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.esp = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.esp)
        norm_x = self.scale * norm_x + self.shift
        return norm_x

def test_LN():
    torch.manual_seed(0)
    batch_size, seq_len, emb_dim = 1, 3, 2
    x = torch.randn(batch_size, seq_len, emb_dim)
    print(f"{x.shape=}\n{x=}")
    ln = LayerNorm(emb_dim)
    print(f"{ln=}")
    output = ln(x)
    print(f"{output=}")

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

# TODO: test for GELU
def test_GELU():
    pass    

# TODO: Swish


class FF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MHA(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            ctx_len=cfg["ctx_len"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        self.ff = FF(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.dropout = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        # arch based on updated transformer popularised by GPT-2
        residual = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x += residual
        
        return x

# %% 
# TODO: GPT Model
class GPT_modal(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg['ctx_len'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_idx):
        '''in_idx: Input tensor of tokenized indices with shape [batch_size, seq_len]'''
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)        

        return logits

def test_GPT_modal(cfg = GPT_CONFIG_124M):
    torch.manual_seed(42)

# %%
def generate_text_simple(model, idx, max_new_tokens, ctx_len):
    '''
    > basic sampler: minial greddy decoder for inference
    > TODO: will be used in training loop to periodically print model samples 

    idx [batch, seq_len] : sequence of generated tokens

    '''
    for _ in range(max_new_tokens):

        # trim token array to supported ctx_len
        # trims prfix token, and takes input from n-th token
        idx_ctx = idx[:, -ctx_len:]
        with torch.no_grad(): 
            logits = model(idx_ctx)
        
        # slices logits: [batch, seq_len, vocab_size] -> [batch, vocab_size]
        # why? 
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        # append new token to existing seq
        idx = torch.cat((idx, idx_next), dim=1) 

    return idx

def test_decoding():
    pass

# %%
def main():
    torch.manual_seed(42)
    start_context = "Hello, I am"
    
    model = GPT_modal(GPT_CONFIG_124M)
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)         # adding batch_dim = 1

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print(f"{start_context=}")    
    print(f"{encoded=}")
    print(f"{encoded_tensor.shape}")

    out = generate_text_simple(
        model = model,
        idx = encoded_tensor,
        max_new_tokens=10,
        ctx_len=GPT_CONFIG_124M["ctx_len"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}") 
    print(f"{out=}")
    print(f"{len(out[0])=}")
    print(f"{decoded_text=}")

if __name__ == "__main__":
    main()