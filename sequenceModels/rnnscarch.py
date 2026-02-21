import numpy as np

word_to_ix = {"hello": 0, "world": 1, "love": 2, "machine": 3, "learning": 4}
ix_to_word = {v: k for k, v in word_to_ix.items()}
data = [[0, 1], [2, 3, 4]]

vocab_size = len(word_to_ix)
hidden_size = 8
lr = 0.1

Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output

bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# Forward + Backpropagation (1 step, single example)
def rnn_step(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0

    # FORWARD PASS
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = softmax(ys[t])
        loss += -np.log(ps[t][targets[t], 0])

    # BACKWARD PASS: initialize gradients as zero
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dh_next = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1  # backprop into y
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dh_next
        dhraw = (1 - hs[t] ** 2) * dh  # backprop through tanh
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t - 1].T)
        dh_next = np.dot(Whh.T, dhraw)

    for param, dparam in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby]):
        param -= lr * dparam

    return loss, hs[len(inputs) - 1]


for epoch in range(1000):
    for seq in data:
        inputs = seq[:-1]
        targets = seq[1:]
        hprev = np.zeros((hidden_size, 1))
        loss, hprev = rnn_step(inputs, targets, hprev)
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, loss: {loss}")


def predict(inputs_seq):
    h = np.zeros((hidden_size, 1))
    for idx in inputs_seq:
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = softmax(y)
    return np.argmax(p)


test_in = [word_to_ix["love"], word_to_ix["machine"]]
predicted_ix = predict(test_in)
print("Predicted next word:", ix_to_word[predicted_ix])
