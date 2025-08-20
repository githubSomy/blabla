# miniGPT_upgraded.py
# Tiny, readable GPT you can train on your own train_data.txt
# Upgrades: GELU, dropout (attn+residual), Pre-LN blocks, top-k+temperature sampling,
# AdamW with weight decay, cosine LR schedule with warmup, checkpoints, and word-ish tokenizer.

import os, re, math, json, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Tokenizer (simple & robust) ----------------
# Splits into "word-ish" tokens and keeps punctuation as separate tokens.
# Example: "Hello, world!" -> ["Hello", ",", "world", "!"]
TOKEN_PATTERN = re.compile(r"\w+|\S", re.UNICODE)

def tokenize(text):
    return TOKEN_PATTERN.findall(text)

class Vocab:
    def __init__(self, tokens, min_freq=1):
        freqs = {}
        for t in tokens:
            freqs[t] = freqs.get(t, 0) + 1
        # special tokens
        self.PAD = "<pad>"
        self.UNK = "<unk>"
        all_tokens = [self.PAD, self.UNK]
        all_tokens += [t for t, c in sorted(freqs.items(), key=lambda x: (-x[1], x[0])) if c >= min_freq]
        self.itos = all_tokens
        self.stoi = {t:i for i,t in enumerate(self.itos)}

    @property
    def size(self): return len(self.itos)

    def encode(self, toks):
        return [self.stoi.get(t, self.stoi[self.UNK]) for t in toks]

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids)

# ---------------- Model ----------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.key   = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj  = nn.Linear(n_embd, n_embd, bias=False)

        self.attn_drop  = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # Causal mask for up to block_size
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.size()
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)               # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v                                                                # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)                           # (B, T, C)
        y = self.resid_drop(self.proj(y))                                          # projection + dropout
        return y

class MLP(nn.Module):
    def __init__(self, n_embd, resid_pdrop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),                          # GELU > ReLU for LMs
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    # Pre-LN Transformer block (LayerNorm before sublayers) -> more stable training
    def __init__(self, n_embd, n_head, block_size, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, attn_pdrop, resid_pdrop)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, resid_pdrop)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size=128, n_layer=6, n_head=8, n_embd=512, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(resid_pdrop)

        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, attn_pdrop, resid_pdrop) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, "Sequence length > block_size"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)

        x = self.tok_emb(idx) + self.pos_emb(pos)   # (B, T, C)
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)                       # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.6, top_k=80):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(1e-6, temperature)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

# ---------------- Training ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="train_data.txt")
    p.add_argument("--out_dir", type=str, default="out")
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=512)
    p.add_argument("--attn_drop", type=float, default=0.2)
    p.add_argument("--resid_drop", type=float, default=0.2)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--eval_interval", type=int, default=1000)
    p.add_argument("--eval_iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--sample_every", type=int, default=2000)
    p.add_argument("--sample_prompt", type=str, default="User: hello\nAssistant:")
    args = p.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # Load and tokenize data
    with open(args.data, "r", encoding="utf-8") as f:
        raw_text = f.read()
    tokens = tokenize(raw_text)
    vocab = Vocab(tokens, min_freq=1)
    ids = torch.tensor(vocab.encode(tokens), dtype=torch.long)

    n = int(0.9 * len(ids))
    train_ids, val_ids = ids[:n], ids[n:]

    def get_batch(split):
        data = train_ids if split == "train" else val_ids
        ix = torch.randint(0, len(data) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([data[i:i+args.block_size] for i in ix])
        y = torch.stack([data[i+1:i+1+args.block_size] for i in ix])
        return x.to(device), y.to(device)

    # Model
    model = MiniGPT(
        vocab_size=vocab.size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        attn_pdrop=args.attn_drop,
        resid_pdrop=args.resid_drop
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )

    # Cosine LR with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    @torch.no_grad()
    def estimate(split, iters=None):
        iters = iters or args.eval_iters
        model.eval(); losses = []
        for _ in range(iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    best_val = float("inf")

    for step in range(1, args.steps + 1):
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stability
        optimizer.step()
        scheduler.step()

        if step % args.eval_interval == 0 or step == 1:
            val = estimate("val", args.eval_iters)
            print(f"step {step}  train_loss={loss.item():.3f}  val_loss={val:.3f}")
            if val < best_val:
                best_val = val
                ckpt = {
                    "model": model.state_dict(),
                    "config": {
                        "vocab": vocab.itos,
                        "block_size": args.block_size,
                        "n_layer": args.n_layer,
                        "n_head": args.n_head,
                        "n_embd": args.n_embd,
                        "attn_drop": args.attn_drop,
                        "resid_drop": args.resid_drop
                    }
                }
                torch.save(ckpt, os.path.join(args.out_dir, "ckpt.pt"))
                with open(os.path.join(args.out_dir, "vocab.json"), "w", encoding="utf-8") as vf:
                    json.dump(vocab.itos, vf, ensure_ascii=False)
                print("[*] saved checkpoint")

        if step % args.sample_every == 0:
            prompt = args.sample_prompt
            # encode prompt using same tokenizer+vocab
            p_ids = vocab.encode(tokenize(prompt))
            x = torch.tensor(p_ids, dtype=torch.long, device=device).unsqueeze(0)
            y = model.generate(x, max_new_tokens=200, temperature=0.8, top_k=80)[0].tolist()
            # decode ids back to string
            inv = vocab.itos
            out_tokens = [inv[i] for i in y]
            print("SAMPLE:", "".join([
                (t if re.match(r"\W$", t) else (" " + t)) for t in out_tokens
            ]).strip())

    # Final sample
    prompt = args.sample_prompt
    p_ids = vocab.encode(tokenize(prompt))
    x = torch.tensor(p_ids, dtype=torch.long, device=device).unsqueeze(0)
    y = model.generate(x, max_new_tokens=200, temperature=0.7, top_k=60)[0].tolist()
    out_tokens = [vocab.itos[i] for i in y]
    print("\nFINAL SAMPLE:\n", "".join([
        (t if re.match(r"\W$", t) else (" " + t)) for t in out_tokens
    ]).strip())

if __name__ == "__main__":
    main()
