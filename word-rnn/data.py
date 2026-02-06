# data.py
import torch

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_vocab(text):
    words = text.split()
    vocab = sorted(set(words))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return vocab, word2idx, idx2word

def encode(text, word2idx):
    return [word2idx[w] for w in text.split()]

def one_hot(indices, vocab_size):
    return torch.eye(vocab_size)[indices]

def get_batches(encoded, seq_len, vocab_size):
    for i in range(len(encoded) - seq_len):
        x_idx = encoded[i:i+seq_len]
        y = encoded[i+1:i+seq_len+1]

        x = one_hot(x_idx, vocab_size)
        y = torch.tensor(y)

        yield x, y
