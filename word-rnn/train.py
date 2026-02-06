# train.py
import torch
import torch.nn as nn
import torch.optim as optim

from data import load_text, build_vocab, encode, get_batches
from model import WordRNN
import config


def main():
    # device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.DEVICE == "cuda" else "cpu"
    )

    # load data
    text = load_text(config.DATA_PATH)
    vocab, word2idx, idx2word = build_vocab(text)
    encoded = encode(text, word2idx)

    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    # model
    model = WordRNN(
        vocab_size=vocab_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS
    ).to(device)

    # loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    # training
    model.train()

    for epoch in range(config.EPOCHS):
        total_loss = 0.0
        h = None  # initial hidden state

        for x, y in get_batches(encoded, config.SEQ_LEN, vocab_size):
            x = x.to(device)          # (seq_len, vocab_size)
            y = y.to(device)          # (seq_len)

            optimizer.zero_grad()

            logits, h = model(x, h)
            # logits: (seq_len, 1, vocab_size)

            # detach hidden state
            h = h.detach()

            # reshape for loss
            logits = logits.squeeze(1)   # (seq_len, vocab_size)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(encoded) - config.SEQ_LEN)
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] | Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
