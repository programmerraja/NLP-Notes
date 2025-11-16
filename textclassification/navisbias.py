import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy

from sklearn import metrics
from sklearn import model_selection
from sklearn.datasets import fetch_20newsgroups
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import os

# spaCy upgrade and package installation (Note: Commands like !pip and !python are for notebook environments)
# !pip install -U spacy==3.*
# !python -m spacy download en_core_web_sm
# !python -m spacy info

# --- First pass at building a Naive Bayes model (Initial run with metadata) ---

# Fetch the entire training corpus with metadata (default)
training_corpus = fetch_20newsgroups(subset="train")

print("Training data size: {}".format(len(training_corpus.data)))

# print(training_corpus.target_names)
# print(training_corpus.target)
# print(training_corpus.data[0])

first_doc_label = training_corpus.target[0]
print("Label for this post: {}".format(first_doc_label))
print("Corresponding topic: {}".format(training_corpus.target_names[first_doc_label]))


# Distribution check
# bins, counts = np.unique(training_corpus.target, return_counts=True)
# freq_series = pd.Series(counts/len(training_corpus.data))
# plt.figure(figsize=(12, 8))
# ax = freq_series.plot(kind='bar')
# ax.set_xticklabels(bins, rotation=0)
# plt.show()

# Split into train and validation sets
(
    train_data_unfiltered,
    val_data_unfiltered,
    train_labels_unfiltered,
    val_labels_unfiltered,
) = train_test_split(
    training_corpus.data, training_corpus.target, train_size=0.8, random_state=1
)
print("Training data size: {}".format(len(train_data_unfiltered)))
print("Validation data size: {}".format(len(val_data_unfiltered)))

# Blank spaCy pipeline for simple tokenization
nlp = spacy.blank("en")


# Tokenizer function for initial run
def spacy_tokenizer_simple(doc):
    # Remove punctuation and spaces, filter for alphabetic characters only, return token text.
    return [
        t.text for t in nlp(doc) if not t.is_punct and not t.is_space and t.is_alpha
    ]


# Vectorize
vectorizer_simple = TfidfVectorizer(tokenizer=spacy_tokenizer_simple)
train_feature_vects_unfiltered = vectorizer_simple.fit_transform(train_data_unfiltered)
print(train_feature_vects_unfiltered)
os._exit(1)

# Train the classifier
nb_classifier_unfiltered = MultinomialNB()
nb_classifier_unfiltered.fit(train_feature_vects_unfiltered, train_labels_unfiltered)

# Evaluate on unfiltered training data
train_preds_unfiltered = nb_classifier_unfiltered.predict(
    train_feature_vects_unfiltered
)
print(
    "F1 score on initial training set (unfiltered): {}".format(
        metrics.f1_score(
            train_labels_unfiltered, train_preds_unfiltered, average="macro"
        )
    )
)

# --- Retraining with filtered data ---

# Remove headers, footers, and quotes from training set and resplit.
filtered_training_corpus = fetch_20newsgroups(
    subset="train", remove=("headers", "footers", "quotes")
)
train_data, val_data, train_labels, val_labels = train_test_split(
    filtered_training_corpus.data,
    filtered_training_corpus.target,
    train_size=0.8,
    random_state=1,
)

print("Example of filtered data point:")
print(train_data[0])

# Revectorize our text and retrain our model with the simple tokenizer
vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer_simple)
train_feature_vects = vectorizer.fit_transform(train_data)
nb_classifier = MultinomialNB()
nb_classifier.fit(train_feature_vects, train_labels)

# Recheck F1 score on filtered training data
train_preds = nb_classifier.predict(train_feature_vects)
print(
    "F1 score on filtered training set: {}".format(
        metrics.f1_score(train_labels, train_preds, average="macro")
    )
)

# Check performance on validation set
val_feature_vects = vectorizer.transform(val_data)
val_preds = nb_classifier.predict(val_feature_vects)
print(
    "F1 score on filtered validation set (simple tokenizer): {}".format(
        metrics.f1_score(val_labels, val_preds, average="macro")
    )
)

# Confusion matrix for validation set (simple tokenizer)
fig, ax = plt.subplots(figsize=(15, 15))
disp = ConfusionMatrixDisplay.from_estimator(
    nb_classifier,
    val_feature_vects,
    val_labels,
    normalize="true",
    display_labels=filtered_training_corpus.target_names,
    xticks_rotation="vertical",
    ax=ax,
)
plt.show()

# Classification report (simple tokenizer)
print(
    metrics.classification_report(
        val_labels, val_preds, target_names=filtered_training_corpus.target_names
    )
)

# --- Improving the model (Lemma and Stopword Removal) ---

print("Training data size: {}".format(len(train_data)))
print(
    "Number of training features (simple tokenizer): {}".format(
        len(train_feature_vects[0].toarray().flatten())
    )
)

# Load statistical spaCy model
nlp = spacy.load("en_core_web_sm")
unwanted_pipes = ["ner", "parser"]


# Tokenizer function for improvement (lemma, stopword removal)
def spacy_tokenizer_improved(doc):
    with nlp.disable_pipes(*unwanted_pipes):
        # Return lemma, removing punctuation, spaces, stopwords, and non-alpha tokens
        return [
            t.lemma_
            for t in nlp(doc)
            if not t.is_punct and not t.is_space and not t.is_stop and t.is_alpha
        ]


# Re-vectorize with the new tokenizer (Improved features)
vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer_improved)
train_feature_vects = vectorizer.fit_transform(train_data)

print(
    "Number of training features (improved tokenizer): {}".format(
        len(train_feature_vects[0].toarray().flatten())
    )
)

# Retrain the classifier
nb_classifier.fit(train_feature_vects, train_labels)
train_preds = nb_classifier.predict(train_feature_vects)
print(
    "Training F1 score with fewer features: {}".format(
        metrics.f1_score(train_labels, train_preds, average="macro")
    )
)

# Check performance on validation set (Improved features)
val_feature_vects = vectorizer.transform(val_data)
val_preds = nb_classifier.predict(val_feature_vects)
print(
    "Validation F1 score with fewer features: {}".format(
        metrics.f1_score(val_labels, val_preds, average="macro")
    )
)

# Confusion matrix for validation set (Improved features)
fig, ax = plt.subplots(figsize=(15, 15))
disp = ConfusionMatrixDisplay.from_estimator(
    nb_classifier,
    val_feature_vects,
    val_labels,
    normalize="true",
    display_labels=filtered_training_corpus.target_names,
    xticks_rotation="vertical",
    ax=ax,
)
plt.show()

# Classification report (Improved features)
print(
    metrics.classification_report(
        val_labels, val_preds, target_names=filtered_training_corpus.target_names
    )
)

# --- Hyperparameter Tuning with Grid Search and Cross-Validation ---

# The alpha values to try.
params = {
    "alpha": [
        0.01,
        0.1,
        0.5,
        1.0,
        10.0,
    ],
}

# Instantiate the search with the model and fit it on the training data.
multinomial_nb_grid = model_selection.GridSearchCV(
    MultinomialNB(), param_grid=params, scoring="f1_macro", n_jobs=-1, cv=5, verbose=2
)
multinomial_nb_grid.fit(train_feature_vects, train_labels)

# Best parameter value(s)
print("Best parameter value(s): {}".format(multinomial_nb_grid.best_params_))

# Use the best estimator on the validation set.
best_nb_classifier = multinomial_nb_grid.best_estimator_
val_preds = best_nb_classifier.predict(val_feature_vects)
print(
    "Validation F1 score after tuning alpha: {}".format(
        metrics.f1_score(val_labels, val_preds, average="macro")
    )
)

# Confusion matrix for validation set (Tuned classifier)
fig, ax = plt.subplots(figsize=(15, 15))
disp = ConfusionMatrixDisplay.from_estimator(
    best_nb_classifier,
    val_feature_vects,
    val_labels,
    normalize="true",
    display_labels=filtered_training_corpus.target_names,
    xticks_rotation="vertical",
    ax=ax,
)
plt.show()

# Classification report (Tuned classifier)
print(
    metrics.classification_report(
        val_labels, val_preds, target_names=filtered_training_corpus.target_names
    )
)

# --- Utility Function: Show Top Words ---


def show_top_words(classifier, vectorizer, categories, top_n):
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    for i, category in enumerate(categories):
        # Sort log probabilities to find the most probable features
        prob_sorted = classifier.feature_log_prob_[i, :].argsort()[::-1]
        print("%s: %s" % (category, " ".join(feature_names[prob_sorted[:top_n]])))


show_top_words(
    best_nb_classifier, vectorizer, filtered_training_corpus.target_names, 10
)

# --- Sanity Check: Dummy Classifier ---

# Train a dummy classifier which just guesses the most frequent class.
dummy_clf_freq = DummyClassifier(strategy="most_frequent")
dummy_clf_freq.fit(train_feature_vects, train_labels)
print(
    f"Dummy Classifier (most_frequent) score: {dummy_clf_freq.score(val_feature_vects, val_labels)}"
)

# Train a dummy classifier which just guesses a class randomly.
dummy_clf_uniform = DummyClassifier(strategy="uniform")
dummy_clf_uniform.fit(train_feature_vects, train_labels)
print(
    f"Dummy Classifier (uniform) score: {dummy_clf_uniform.score(val_feature_vects, val_labels)}"
)

# --- Creating the final Naive Bayes classifier Pipeline ---

# We'll use the entire original training set (including validation data) and the ideal alpha param.
text_classifier = Pipeline(
    [
        ("vectorizer", TfidfVectorizer(tokenizer=spacy_tokenizer_improved)),
        ("classifier", MultinomialNB(alpha=0.01)),  # best alpha found by grid search
    ]
)

text_classifier.fit(filtered_training_corpus.data, filtered_training_corpus.target)

# Download the 20 newsgroups test dataset.
filtered_test_corpus = fetch_20newsgroups(
    subset="test", remove=("headers", "footers", "quotes")
)

# Predict on the raw test data
test_preds = text_classifier.predict(filtered_test_corpus.data)

# Confusion Matrix for Test Data
fig, ax = plt.subplots(figsize=(15, 15))
ConfusionMatrixDisplay.from_predictions(
    filtered_test_corpus.target,
    test_preds,
    normalize="true",
    display_labels=filtered_test_corpus.target_names,
    xticks_rotation="vertical",
    ax=ax,
)
plt.show()

# Classification report for Test Data
print(
    metrics.classification_report(
        filtered_test_corpus.target,
        test_preds,
        target_names=filtered_test_corpus.target_names,
    )
)

# --- Utility Function: Classify New Text ---


def classify_text(clf, doc, labels=None):
    # Get probability distribution for the document
    probas = clf.predict_proba([doc]).flatten()
    # Find the index of the class with the maximum probability
    max_proba_idx = np.argmax(probas)

    if labels:
        most_proba_class = labels[max_proba_idx]
    else:
        most_proba_class = max_proba_idx
    return (most_proba_class, probas[max_proba_idx])


# Test cases
# Post from r/medicine.
s1 = "Hello everyone so am doing my thesis on Ischemic heart disease have been using online articles and textbooks mostly Harrisons internal med. could u recommended me some source specifically books where i can get more about in depth knowledge on IHD."
print(
    f"'{s1[:30]}...': {classify_text(text_classifier, s1, filtered_test_corpus.target_names)}"
)

# Post from r/space.
s2 = "First evidence that water can be created on the lunar surface by Earth's magnetosphere. Particles from Earth can seed the moon with water, implying that other planets could also contribute water to their satellites."
print(
    f"'{s2[:30]}...': {classify_text(text_classifier, s2, filtered_test_corpus.target_names)}"
)

# Post from r/cars.
s3 = "New Toyota 86 Launch Reportedly Delayed to 2022, CEO Doesn't Want a Subaru Copy"
print(
    f"'{s3[:30]}...': {classify_text(text_classifier, s3, filtered_test_corpus.target_names)}"
)

# Post from r/electronics.
s4 = "My First Ever Homemade PCB. My SMD Soldering Skills Aren't Great, But I'm Quite Proud of it."
print(
    f"'{s4[:30]}...': {classify_text(text_classifier, s4, filtered_test_corpus.target_names)}"
)

# Made-up statements with low probability
s5 = "I don't know if that's a good idea."
print(
    f"'{s5[:30]}...': {classify_text(text_classifier, s5, filtered_test_corpus.target_names)}"
)

s6 = "Hold on for dear life."
print(
    f"'{s6[:30]}...': {classify_text(text_classifier, s6, filtered_test_corpus.target_names)}"
)
