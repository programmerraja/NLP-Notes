from keras.models import Sequential
import tensorflow as tf
from keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

corpus = [
    "I love machine learning",
    "machine learning is amazing",
    "I love deep learning",
    "deep learning with RNN is powerful",
    "RNN models can predict next word",
    "TensorFlow makes building models easier",
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
# build a  word_index  look like: {'<OOV>': 1, 'the': 2, 'cat': 3, 'sat': 4, 'on': 5, 'mat': 6}


sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[: i + 1]
        sequences.append(n_gram_sequence)

max_seq_len = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_seq_len, padding="pre")

sequences = np.array(sequences)


X, y = sequences[:, :-1], sequences[:, -1]
print(sequences, "tensorflow")
print("Padded Sequences:\n", X, y)
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 10

model = Sequential(
    [
        Embedding(vocab_size, embedding_dim, input_length=max_seq_len - 1),
        SimpleRNN(50),
        Dense(vocab_size, activation="softmax"),
    ]
)

model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# Train the model
model.fit(X, y, epochs=100, verbose=1)


# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_seq_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding="pre")
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""


# Example usage
input_text = "machine"
predicted_word = predict_next_word(model, tokenizer, input_text, max_seq_len)
print(f"Next word prediction after '{input_text}': {predicted_word}")
