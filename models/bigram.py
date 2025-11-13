import random
from collections import defaultdict


def prepare_data():
    sentences = [
        "The cat sat on the mat",
        "The dog barked at the cat",
        "The bird sang a song",
        "The cat chased the mouse",
        "The dog and the bird played together",
    ]
    tokenized_sentences = [sentence.split() for sentence in sentences]
    return tokenized_sentences


def build_bigram_model(sentences):
    bigram_counts = defaultdict(int)
    unigram_counts = defaultdict(int)

    for sentence in sentences:
        for i in range(len(sentence) - 1):
            bigram = (sentence[i], sentence[i + 1])
            bigram_counts[bigram] += 1
            unigram_counts[sentence[i]] += 1
        unigram_counts[sentence[-1]] += 1

    bigram_probabilities = {}
    for bigram, count in bigram_counts.items():
        bigram_probabilities[bigram] = count / unigram_counts[bigram[0]]

    return bigram_probabilities


# Step 3: Generate text using the bigram model
def generate_text(bigram_probabilities, start_word, num_words=10):
    current_word = start_word
    generated_text = [current_word]

    for _ in range(num_words - 1):
        # Filter bigrams that start with the current word
        candidates = {
            bigram: prob
            for bigram, prob in bigram_probabilities.items()
            if bigram[0] == current_word
        }
        if not candidates:
            break  # Stop if no valid bigram is found

        # Choose the next word based on probabilities
        next_word = random.choices(
            list(candidates.keys()), weights=list(candidates.values())
        )[0][1]
        generated_text.append(next_word)
        current_word = next_word

    return " ".join(generated_text)


# Main execution
if __name__ == "__main__":
    sentences = prepare_data()
    bigram_probabilities = build_bigram_model(sentences)

    print("Bigram Probabilities:")
    for bigram, prob in bigram_probabilities.items():
        print(f"{bigram}: {prob:.2f}")

    print("\nGenerated Text:")
    start_word = "The"
    generated_text = generate_text(bigram_probabilities, start_word)
    print(generated_text)
