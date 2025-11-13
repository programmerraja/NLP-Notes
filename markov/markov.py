import random
from collections import defaultdict

# Sample training text
text = """
the cat sat on the mat and the cat ate a rat the rat ran away from the cat
"""

# STEP 1: Tokenize (split into words)
words = text.strip().split()

# STEP 2: Build transition probabilities
transitions = defaultdict(list)
for i in range(len(words) - 1):
    curr_word = words[i]
    next_word = words[i + 1]
    transitions[curr_word].append(next_word)

print(transitions)


# STEP 3: Generate new text
def generate_text(start_word, length=15):
    word = start_word
    output = [word]
    for _ in range(length - 1):
        if word not in transitions:
            break
        word = random.choice(transitions[word])
        output.append(word)
    return " ".join(output)


# Example usage
print(generate_text("the", 20))
