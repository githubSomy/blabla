import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# ====== Load dataset ======
with open("train_data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# --- Tokenize at word level ---
words = text.split()
vocab = sorted(list(set(words)))
stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for w, i in stoi.items()}

encode = lambda s: [stoi[w] for w in s.split() if w in stoi]   # string -> list of ints
decode = lambda l: " ".join([itos[i] for i in l])              # list -> string

data = torch.tensor(encode(text), dtype=torch.long)

# Train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ====== Hyperparameters ======
block_size = 128
batch_size = 32
max_iters = 2000   # adjust for 10–20 mins training
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 512
n_head = 4
n_layer = 4

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ====== Model ======
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(len(vocab), n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, len(vocab))

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits = logits.view(B*T, -1)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ====== Train ======
model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters+1):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        print(f"step {iter}: train loss {loss.item()}")

# ====== Interactive Chat ======
def chat():
    print("\nChatbot ready! Type 'quit' to exit, or 'Training data' to test accuracy.\n")
    while True:
        user_inp = input("User: ")
        if user_inp.lower() == "quit":
            break

        if user_inp == "Training data":
            correct, total = 0, 0
            for line in text.splitlines():
                if line.startswith("User: "):
                    q = line.replace("User: ", "")
                if line.startswith("Bot: "):
                    gold = line.replace("Bot: ", "")
                    total += 1
                    start = time.time()
                    context = torch.tensor([encode(q)], dtype=torch.long, device=device)
                    out = model.generate(context, max_new_tokens=20)
                    bot_reply = decode(out[0].tolist())[len(q.split()):]
                    elapsed = time.time() - start
                    if gold.split()[0] in bot_reply:
                        correct += 1
                    print(f"Q: {q}\nExpected: {gold}\nGot: {bot_reply}\nTime: {elapsed:.2f}s\n")
            print(f"Accuracy: {correct}/{total} = {100*correct/total:.2f}%")
            continue

        # normal conversation
        start = time.time()
        context = torch.tensor([encode(user_inp)], dtype=torch.long, device=device)
        out = model.generate(context, max_new_tokens=20)
        bot_reply = decode(out[0].tolist())[len(user_inp.split()):]
        elapsed = time.time() - start
        print(f"Bot: {bot_reply} (responded in {elapsed:.2f}s)")

if __name__ == "__main__":
    chat()
