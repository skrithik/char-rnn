# Training logic
from rnn_cell import SimpleRNN, softmax
import math
# STEP 1: Read raw text data

with open("data.txt", "r") as f:
    text = f.read()

# print("Raw text:")
# print(text)


# STEP 2: Build vocabulary (unique characters)

chars = sorted(list(set(text)))

# print("\nUnique characters:")
# print(chars)
# print("Vocabulary size:", len(chars))


# STEP 3: Character to index mapping

char_to_ix = {}
ix_to_char = {}

for i, ch in enumerate(chars):
    char_to_ix[ch] = i
    ix_to_char[i] = ch

# print("\nCharacter to index mapping:")
# print(char_to_ix)


# STEP 4: Create training pairs

inputs = []
targets = []

for i in range(len(text) - 1):
    inputs.append(char_to_ix[text[i]])
    targets.append(char_to_ix[text[i + 1]])

# print("\nFirst 10 input-target pairs:")
# for i in range(10):
#     print(text[i], "->", text[i+1])

#one-hot encoding
def one_hot(ix, vocab_size):
    v = [0.0] * vocab_size
    v[ix] = 1.0
    return v

inputs_oh = [one_hot(ix, len(chars)) for ix in inputs]


rnn = SimpleRNN(len(chars))


#loss function

def cross_entropy_loss(y_probs, target_ix):
    """
    y_probs: softmax output (list of probabilities)
    target_ix: correct character index
    """
    return -math.log(y_probs[target_ix])


# total_loss = 0.0

# print("\nLoss computation:\n")

# for t in range(10):
#     x_t = inputs_oh[t]
#     y_true = targets[t]

#     y_logits = rnn.forward(x_t)
#     y_probs = softmax(y_logits)

#     loss_t = cross_entropy_loss(y_probs, y_true)
#     total_loss += loss_t

#     print(
#         f"t={t+1} | true='{ix_to_char[y_true]}' | loss={loss_t:.4f}"
#     )

# print("\nTotal loss (first 10 steps):", total_loss)


# learning_rate = 0.1
# epochs = 20

# from rnn_cell import SimpleRNN, softmax
# import math

# rnn = SimpleRNN(len(chars))

# for epoch in range(epochs):
#     total_loss = 0.0
#     rnn.h = 0.0  # reset hidden state each epoch

#     for t in range(len(inputs_oh)):
#         x_t = inputs_oh[t]
#         y_true = targets[t]

#         # forward
#         y_logits = rnn.forward(x_t)
#         y_probs = softmax(y_logits)

#         # loss
#         loss = -math.log(y_probs[y_true])
#         total_loss += loss

#         # ---- BACKPROP (output layer only) ----
#         for k in range(len(chars)):
#             y_k = 1.0 if k == y_true else 0.0
#             error = y_probs[k] - y_k

#             # gradients
#             dWy = error * rnn.h
#             dby = error

#             # update
#             rnn.Wy[k] -= learning_rate * dWy
#             rnn.by[k] -= learning_rate * dby

#     print(f"Epoch {epoch+1}, Loss: {total_loss:.3f}")
learning_rate = 0.05
epochs = 30000


rnn = SimpleRNN(len(chars))

for epoch in range(epochs):
    # ---- FORWARD PASS ----
    rnn.h = 0.0
    hs = []          # a⟨t⟩
    zs = []          # z⟨t⟩
    ys = []          # ŷ⟨t⟩

    total_loss = 0.0

    for t in range(len(inputs_oh)):
        x_t = inputs_oh[t]
        y_true = targets[t]

        # compute z⟨t⟩
        wx_dot_x = sum(rnn.Wx[i] * x_t[i] for i in range(len(x_t)))
        z_t = wx_dot_x + rnn.Wh * rnn.h + rnn.b

        a_t = math.tanh(z_t)
        rnn.h = a_t

        # output
        y_logits = [rnn.Wy[k] * a_t + rnn.by[k] for k in range(len(chars))]
        y_probs = softmax(y_logits)

        loss = -math.log(y_probs[y_true])
        total_loss += loss

        zs.append(z_t)
        hs.append(a_t)
        ys.append(y_probs)

    # ---- BACKWARD PASS (BPTT) ----
    dWx = [0.0] * len(rnn.Wx)
    dWh = 0.0
    db  = 0.0
    dWy = [0.0] * len(rnn.Wy)
    dby = [0.0] * len(rnn.by)

    da_next = 0.0

    for t in reversed(range(len(inputs_oh))):
        y_true = targets[t]
        y_probs = ys[t]
        a_t = hs[t]
        a_prev = hs[t-1] if t > 0 else 0.0
        x_t = inputs_oh[t]

        # output gradients
        for k in range(len(chars)):
            y_k = 1.0 if k == y_true else 0.0
            dy = y_probs[k] - y_k
            dWy[k] += dy * a_t
            dby[k] += dy

        # hidden gradient
        da = sum((y_probs[k] - (1.0 if k == y_true else 0.0)) * rnn.Wy[k]
                 for k in range(len(chars)))
        da += da_next

        dz = da * (1 - a_t * a_t)

        for i in range(len(x_t)):
            dWx[i] += dz * x_t[i]

        dWh += dz * a_prev
        db  += dz

        da_next = dz * rnn.Wh

    # ---- UPDATE ----
    for i in range(len(rnn.Wx)):
        rnn.Wx[i] -= learning_rate * dWx[i]

    rnn.Wh -= learning_rate * dWh
    rnn.b  -= learning_rate * db

    for k in range(len(chars)):
        rnn.Wy[k] -= learning_rate * dWy[k]
        rnn.by[k] -= learning_rate * dby[k]

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss:.3f}")

def predict_next_char(rnn, char, char_to_ix, ix_to_char):
    """
    Predict next character given a single input character
    """
    # reset hidden state
    rnn.h = 0.0

    # one-hot encode input
    x = [0.0] * len(char_to_ix)
    x[char_to_ix[char]] = 1.0

    # forward pass
    y_logits = rnn.forward(x)
    y_probs = softmax(y_logits)

    # pick most probable character
    pred_ix = y_probs.index(max(y_probs))
    return ix_to_char[pred_ix]


print("\n--- Prediction mode (type 'exit' to quit) ---")

while True:
    user_input = input("Enter a character: ")

    if user_input == "exit":
        break

    if user_input not in char_to_ix:
        print("Character not in vocabulary!")
        continue

    pred = predict_next_char(rnn, user_input, char_to_ix, ix_to_char)
    print(f"Predicted next character: '{pred}'")
