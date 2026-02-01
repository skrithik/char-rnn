# RNN cell logic
import math
import random

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def softmax(xs):
    exps = [math.exp(x) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

class SimpleRNN:
    def __init__(self, vocab_size):
        # n_a = 1 (one hidden unit)

        # Andrew: Wax (1 × |V|)
        self.Wx = [random.uniform(-0.5, 0.5) for _ in range(vocab_size)]

        # Andrew: Waa (scalar)
        self.Wh = random.uniform(-0.5, 0.5)

        # Andrew: ba
        self.b = 0.0

        # Andrew: Wya (|V| × 1)
        self.Wy = [random.uniform(-0.5, 0.5) for _ in range(vocab_size)]
        self.by = [0.0 for _ in range(vocab_size)]

        # a⟨0⟩
        self.h = 0.0

    def forward(self, x_vec):
        """
        x_vec = x⟨t⟩ one-hot vector
        """

        # Wax x⟨t⟩  (dot product)
        wx_dot_x = 0.0
        for i in range(len(x_vec)):
            wx_dot_x += self.Wx[i] * x_vec[i]

        # a⟨t⟩
        self.h = tanh(wx_dot_x + self.Wh * self.h + self.b)

        # y⟨t⟩ logits
        y = []
        for k in range(len(self.Wy)):
            y.append(self.Wy[k] * self.h + self.by[k])

        return y

