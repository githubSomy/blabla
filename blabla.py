# chatbot.py
# Chat with your trained miniGPT in single-turn replies.

import json, re, torch
from miniGPT import MiniGPT, tokenize  # uses your upgraded training code

def load_model(ckpt_path="out/ckpt.pt", vocab_path="out/vocab.json", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]

    with open(vocab_path, "r", encoding="utf-8") as f:
        itos = json.load(f)  # exact vocab from training (list of tokens)
    stoi = {t: i for i, t in enumerate(itos)}

    model = MiniGPT(
        vocab_size=len(itos),
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        attn_pdrop=config["attn_drop"],
        resid_pdrop=config["resid_drop"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ids for control
    tok_User = stoi.get("User", None)
    tok_Bot  = stoi.get("Bot", None)
    tok_colon = stoi.get(":", None)
    if None in (tok_User, tok_Bot, tok_colon):
        raise RuntimeError("Your vocab is missing 'User', 'Bot' or ':'. Keep dataset role tags consistent.")

    # ban specials from generation
    ban_ids = set()
    if "<unk>" in stoi: ban_ids.add(stoi["<unk>"])
    if "<pad>" in stoi: ban_ids.add(stoi["<pad>"])

    return model, (itos, stoi), device, tok_User, tok_Bot, tok_colon, ban_ids

def detok(itos, ids):
    # Nicely join tokens: add space before word-ish tokens, no space before pure punctuation
    out = []
    for t in (itos[i] for i in ids):
        if re.match(r"^\W+$", t):  # punctuation-like token
            out.append(t)
        else:
            if out and not out[-1].endswith(" "):
                out.append(" ")
            out.append(t)
    return "".join(out).strip()

@torch.no_grad()
def generate_reply(model, itos, stoi, device, ctx_ids, block_size, stop_bigram, ban_ids,
                   max_new_tokens=120, temperature=0.7, top_k=60):
    ids = torch.tensor([ctx_ids[-block_size:]], dtype=torch.long, device=device)
    generated = []

    for _ in range(max_new_tokens):
        idx_cond = ids[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(1e-6, temperature)

        # ban tokens
        if ban_ids:
            logits[:, list(ban_ids)] = float("-inf")

        # top-k
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        # check if all -inf (no valid tokens left)
        if torch.all(~torch.isfinite(logits)):
            # fallback: uniform over all *non-banned* tokens
            mask = torch.ones_like(logits)
            if ban_ids:
                mask[:, list(ban_ids)] = 0
            probs = mask / mask.sum()
        else:
            probs = torch.softmax(logits, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1)
        next_id_int = int(next_id.item())
        ids = torch.cat([ids, next_id], dim=1)
        generated.append(next_id_int)

        # stop at "User:" bigram
        if len(generated) >= 2:
            if generated[-2] == stop_bigram[0] and generated[-1] == stop_bigram[1]:
                generated = generated[:-2]
                break

    new_ctx = ctx_ids + generated
    return new_ctx, generated

def chat():
    model, (itos, stoi), device, tok_User, tok_Bot, tok_colon, ban_ids = load_model()
    block_size = model.block_size

    print("🤖 MiniGPT Chatbot (type 'quit' to exit)\n")

    # Conversation context in token ids
    ctx_ids = []

    while True:
        user_in = input("You: ").strip()
        if user_in.lower() in ("quit", "exit"): break

        # Append: User : <tokens>  Bot :
        user_toks = ["User", ":", *tokenize(user_in), "Bot", ":"]
        ctx_ids.extend(stoi.get(t, stoi.get("<unk>", 1)) for t in user_toks)

        # Generate only the Bot reply, and stop when next 'User' ':' appears
        stop_bigram = (tok_User, tok_colon)
        ctx_ids, reply_ids = generate_reply(
            model, itos, stoi, device, ctx_ids, block_size, stop_bigram, ban_ids,
            max_new_tokens=120, temperature=0.65, top_k=50
        )

        # Decode reply tokens to text
        reply_text = detok(itos, reply_ids)
        print(f"Bot: {reply_text}\n")

        # After printing, also append a separator turn to keep structure tight (optional)
        # (No-op here because dataset has no explicit newline token)

if __name__ == "__main__":
    chat()
