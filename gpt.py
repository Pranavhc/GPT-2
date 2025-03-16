from dataclasses import dataclass
import math
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch_directml as dml # type: ignore 
import tiktoken

@dataclass
class GPTConfig:
    # these match the GPT-2 124M model
    block_size: int = 1024      # Coontext length for training
    vocab_size: int = 50257     # number of tokens in the vocabulary
    n_layer: int = 12           # number of transformer layers
    n_head: int = 12            # number of heads in the multi-head attention
    n_embd: int = 768           # embedding dimension


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        assert config.n_embd % config.n_head == 0 # n_embd must be divisible by n_head
        # key, query, value projections for all heads in a single batch
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # a flag that we added to scale the initialization of the weights # type: ignore 

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # causal mask to ensure that attention is only applied to the left in the input sequence, calling it bias because that's what OpenAI called it in their implementation
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension

        # nh is "number of heads", hs is "head size", and C (number of channels) = nh*hs
        # E.g. in GPT-2 (124m), n_head = 12, hs = 64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2) # split the output of the linear projection into q, k, v

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)

        # attention 
        # att = (q@k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)]

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # same as the above code but faster -> uses Flash Attention.

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # a flag that we added to scale the initialization of the weights # type: ignore 


    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig): 
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme 
        self.transformer.wte.weight = self.lm_head.weight # weight object is shared, not just the data. So changing one will change the other.
        # why did we do that? 
        # First of all, we saved a lot of memory. 38597376 floating point numbers (parameters) are now just not needed, about 300MB of memory saved.
        # Sacond, the input token embedding and the output are two ways to look at the same thing. They both are a different representation of the same vocabulary.
        # but they don't have to be different. They can be the same. So we can share the weights between the two.
        # This increases Generalization and reduces redundancy, because the model is forced to learn a single embedding that is good for both input and output.
        # output embedding aren't exactly the same as input embeddings, but they are close enough to go with this analogy. 

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."

        # forward the positional and token embeddings
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) #  positional embeddings of shape (B, T, n_embds)
        token_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = token_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # logits are being viwed as (B*T, vocab_size) and targets as (B*T)
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model from Huggingface"""

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        # following layers were transposed in the original model, we don't want that
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all the candidate parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters()} 
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} 

        # create optim groups. Any parameter that is 2D will be weight decayed, otherwise not.
        # i.e. all weight tensors in matmuls + embeddings will decay, all bias and norm parameters will not.
        decay_params = [p for n, p in param_dict.items() if p.dim() >=2] # only decay the weights that participate in the matmuls (includes embeddings)
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)} with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)} with {num_nodecay_params:,} parameters")

        # create the optimizer itself and use the fused version if it is available
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)
        return optimizer

def load_tokens(filename):
    npt = np.load(filename)
    ppt = torch.tensor(npt, dtype=torch.long)
    return ppt

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B, self.T = B, T
        assert split in {'train', 'val'}

        data_root = 'edu_fineweb10B'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no {split} shards found in {data_root}"

        self.current_shard = 0
        self.tokens = load_tokens(shards[self.current_shard])
        self.current_position = self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B*T

        if self.current_position+(B*T+1) > len(self.tokens): 
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B*T
        return x, y
    
# ------------------------------------------------------------------------

device = dml.device()

train_loader = DataLoaderLite(B=4, T=512, split='train') # GPT2 in reality uses T=1024, my VRAM is limited (12GB) so I'm using T=512

# make computations faster by using TensorFloat32 (TF32) precision
torch.set_float32_matmul_precision('high') # default is 'highest' which is float32 and 'high' is TF32 which is less accurate but faster
# but this line has no effect with my RX 7700XT, might work with NVIDIA GPUs

# model = GPT.from_pretrained('gpt2') # pretained model from Huggingface
model = GPT(GPTConfig(vocab_size=50304)) # we set vocab_size 50304 from 50257 to have a number that is better for the GPU. Parallelism is better with powers of 2. Definitely made a difference in my case. (-250ms per step)
model.to(device)

# model = torch.compile(model) # doesn't work with DirectML

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073

def get_lr(step):
    # 1. Linear warmup 
    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps
    
    # 2. after max_steps, turn down the learning rate
    if step > max_steps:
        return min_lr
    
    # 3. in between, use cosine decay down to min_lr
    decay_ration = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ration <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ration)) # starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def train():
    model.train()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8) # Adam and AdamW don't work with my version of DirectML
    # optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=3e-4) # basic RMSprop works better in my case over the custom configured one

    for step in range(max_steps):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip gradients to prevent explosion
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)

        print(f"Step {step+1}, Loss: {loss.item():.8f}, lr: {lr:.4f} norm: {norm:.4f}, Time: {dt:.2f}ms, Tokens/sec: {tokens_per_sec:.2f}")

        if (step+1) % 1000 == 0: # save model every 1000 steps
            torch.save(model.state_dict(), f'models/gpt2_model_{step+1}.pth')
            print("Model saved")
    
    torch.save(model.state_dict(), 'models/gpt2_model.pth')
# train()

def generate():
    model.load_state_dict(torch.load('models/gpt2_model_1000.pth', weights_only=False))
    model.eval()
    
    num_return_sequences = 1
    max_length = 300

    enc = tiktoken.get_encoding('gpt2')
    new_tokens = enc.encode("The capital of France is")
    new_tokens = torch.tensor(new_tokens, dtype=torch.long) # (1, T)
    new_tokens = new_tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (B, T)
    x = new_tokens.to(device)

    temperature = 1.5
    while x.size(1) < max_length:
        with torch.no_grad():
            logits, loss = model(x) # (B, T, vocab_size)
            # take the logits at the last position 
            logits = logits[:, -1, :] # (B, vocab_size)
            probs = F.softmax(logits/temperature, dim=-1)

            # topk sampling - sample from the top k most likely words
            # topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from top-k probabilities
            # ix = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
            # gather corresponding token indices
            # xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            ix = torch.multinomial(probs, num_samples=1) # (B, 1) # top_k seems to give me the same token for all 5 sequences
            x = torch.cat((x, ix), dim=1)

    # print generated text
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"\nGenerated {i+1}: {decoded}")
generate()


# 1:41:00 to 1:44:00 for first 100 steps
# 1:41:00 to 2:3:00 for first 1000 steps