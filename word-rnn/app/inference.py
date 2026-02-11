import os
import torch
from app.model import WordRNN
from app.data import one_hot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
word2idx = None
idx2word = None
vocab_size = None


def load_model():
    global model, word2idx, idx2word, vocab_size

    # Get project root directory safely
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_path = os.path.join(BASE_DIR, "checkpoints", "word_rnn.pt")

    print("Loading checkpoint from:", ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    word2idx = ckpt["word2idx"]
    idx2word = ckpt["idx2word"]
    vocab_size = len(ckpt["vocab"])

    model = WordRNN(
        vocab_size=vocab_size,
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("Model loaded successfully.")


@torch.no_grad()
def generate(prompt: str, temperature=1.0, max_len=30):
    global model, word2idx, idx2word, vocab_size

    if model is None:
        return "Model not loaded."

    start_words = prompt.split()
    indices = [word2idx[w] for w in start_words if w in word2idx]

    if len(indices) == 0:
        return "Words not in vocabulary."

    h = None
    result = start_words[:]

    # Warm up
    for idx in indices:
        x = one_hot([idx], vocab_size).to(DEVICE)
        _, h = model(x, h)
        h = h.detach()

    current_idx = indices[-1]

    for _ in range(max_len):
        x = one_hot([current_idx], vocab_size).to(DEVICE)
        logits, h = model(x, h)
        h = h.detach()

        logits = logits.squeeze() / temperature
        probs = torch.softmax(logits, dim=0)

        next_idx = torch.multinomial(probs, 1).item()
        next_word = idx2word[next_idx]

        result.append(next_word)
        current_idx = next_idx

    return " ".join(result)
