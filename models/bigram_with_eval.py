import random
from collections import defaultdict
import math

text = "I like apples I like bananas I like apples and bananas"
words = text.split()

transitions = defaultdict(lambda: defaultdict(int))
for i in range(len(words) - 1):
    transitions[words[i]][words[i + 1]] += 1

print(transitions)

# Convert counts → probabilities
model = {}
for cur, nexts in transitions.items():
    total = sum(nexts.values())
    model[cur] = {w: c / total for w, c in nexts.items()}


def compute_perplexity(model, test_sentence):
    test_words = test_sentence.split()
    log_prob_sum = 0
    N = 0

    for i in range(1, len(test_words)):
        prev_word = test_words[i - 1]
        word = test_words[i]

        # Get probability p(word | prev_word)
        prob = model.get(prev_word, {}).get(word, 1e-8)  # small smoothing
        log_prob_sum += math.log(prob)
        N += 1

    avg_log_prob = log_prob_sum / N
    perplexity = math.exp(-avg_log_prob)
    return perplexity


test_sentence = "I like apples and bananas"
pp = compute_perplexity(model, test_sentence)

print("Model transitions:")
for k, v in model.items():
    print(f"{k:10} → {v}")

print("\nTest sentence:", test_sentence)
print("Perplexity:", pp)
