# sample.py
import torch
import config
from model import WordRNN
from data import one_hot


def generate_text(
    model,
    start_words,
    word2idx,
    idx2word,
    vocab_size,
    temperature=1.0,
    max_len=30,
    device="cpu",
):
    model.eval()

    # Convert prompt words to indices
    indices = [word2idx[w] for w in start_words if w in word2idx]
    if len(indices) == 0:
        raise ValueError("Prompt words not in vocabulary.")

    h = None
    result = start_words[:]

    # Warm up model with prompt
    for idx in indices:
        x = one_hot([idx], vocab_size).to(device)
        _, h = model(x, h)
        h = h.detach()

    current_idx = indices[-1]

    # Generate new words
    for _ in range(max_len):
        x = one_hot([current_idx], vocab_size).to(device)
        logits, h = model(x, h)
        h = h.detach()

        logits = logits.squeeze() / temperature
        probs = torch.softmax(logits, dim=0)

        # Top-k sampling
        k = 20
        topk_probs, topk_indices = torch.topk(probs, k)
        topk_probs = topk_probs / torch.sum(topk_probs)

        next_idx = topk_indices[torch.multinomial(topk_probs, 1)].item()
        next_word = idx2word[next_idx]

        result.append(next_word)
        current_idx = next_idx

    return " ".join(result)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.DEVICE == "cuda" else "cpu"
    )

    # Load checkpoint
    ckpt = torch.load("checkpoints/word_rnn.pt", map_location=device)

    vocab = ckpt["vocab"]
    word2idx = ckpt["word2idx"]
    idx2word = ckpt["idx2word"]
    vocab_size = len(vocab)

    # Rebuild model
    model = WordRNN(
        vocab_size=vocab_size,
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])

    # Change this prompt freely
    prompt = "the King".split()

    text = generate_text(
        model,
        prompt,
        word2idx,
        idx2word,
        vocab_size,
        temperature=config.TEMPERATURE,
        max_len=config.MAX_GEN_LEN,
        device=device,
    )

    print("\nGenerated text:\n")
    print(text)


if __name__ == "__main__":
    main()
