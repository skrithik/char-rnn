# model.py
import torch
import torch.nn as nn

class WordRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super().__init__()

        # RNN: one-hot input → hidden state
        self.rnn = nn.RNN(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )

        # Hidden state → vocab logits
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, h_0=None):
        """
        x: (seq_len, vocab_size)
        """

        # add batch dimension
        x = x.unsqueeze(1)  # (seq_len, 1, vocab_size)

        # RNN forward
        out, h_n = self.rnn(x, h_0)
        # out: (seq_len, 1, hidden_size)

        # project each timestep to vocab
        logits = self.fc(out)
        # logits: (seq_len, 1, vocab_size)

        return logits, h_n
