# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim

from data import load_text, build_vocab, encode, get_batches
from model import WordRNN
import config


def main():
    # ================= device =================
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.DEVICE == "cuda" else "cpu"
    )
    print("Using device:", device)

    # ================= data =================
    text = load_text(config.DATA_PATH)
    vocab, word2idx, idx2word = build_vocab(text)
    encoded = encode(text, word2idx)

    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")
    print(f"Total words: {len(encoded)}")

    # ================= model =================
    model = WordRNN(
        vocab_size=vocab_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS
    ).to(device)

    # ================= loss & optimizer =================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    # ================= training =================
    model.train()

    for epoch in range(config.EPOCHS):
        total_loss = 0.0
        step_count = 0
        h = None  # initial hidden state

        print(f"\nEpoch {epoch+1}/{config.EPOCHS} started")

        for i, (x, y) in enumerate(get_batches(encoded, config.SEQ_LEN, vocab_size)):
            x = x.to(device)   # (seq_len, vocab_size)
            y = y.to(device)   # (seq_len)

            optimizer.zero_grad()

            logits, h = model(x, h)

            # truncate BPTT
            h = h.detach()

            logits = logits.squeeze(1)  # (seq_len, vocab_size)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step_count += 1

            # progress print
            if i % 1000 == 0:
                print(f"  Step {i} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / step_count
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] completed | Avg Loss: {avg_loss:.4f}")

    # ================= SAVE MODEL =================
    os.makedirs("checkpoints", exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": vocab,
            "word2idx": word2idx,
            "idx2word": idx2word,
            "hidden_size": config.HIDDEN_SIZE,
            "num_layers": config.NUM_LAYERS,
            "seq_len": config.SEQ_LEN,
        },
        "checkpoints/word_rnn.pt",
    )

    print("\nModel saved to checkpoints/word_rnn.pt")


if __name__ == "__main__":
    main()
