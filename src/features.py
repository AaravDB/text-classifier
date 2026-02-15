# src/features.py

"""
Feature extraction implemented from scratch:

1. Bag of Words
2. TF-IDF

No sklearn vectorizers used.
"""

import numpy as np
import re
from collections import Counter, defaultdict


# ------------------------
# Text preprocessing
# ------------------------

def tokenize(text):
    """
    Converts text into list of words.
    - lowercase
    - removes punctuation
    """

    text = text.lower()

    # keep only words
    words = re.findall(r'\b[a-z]+\b', text)

    return words


# ------------------------
# Vocabulary building
# ------------------------

def build_vocabulary(texts, max_features=None):
    """
    Builds vocabulary from training texts.
    """

    word_counts = Counter()

    for text in texts:
        words = tokenize(text)
        word_counts.update(words)

    # select most common words if max_features specified
    if max_features:
        most_common = word_counts.most_common(max_features)
        vocab = {word: i for i, (word, _) in enumerate(most_common)}
    else:
        vocab = {word: i for i, word in enumerate(word_counts.keys())}

    return vocab


# ------------------------
# Bag of Words
# ------------------------

def bow_transform(texts, vocab):
    """
    Converts texts into Bag of Words matrix.
    """

    X = np.zeros((len(texts), len(vocab)))

    for i, text in enumerate(texts):

        words = tokenize(text)

        counts = Counter(words)

        for word, count in counts.items():
            if word in vocab:
                X[i, vocab[word]] = count

    return X


def get_bow_features(train_texts, test_texts, max_features=None):

    vocab = build_vocabulary(train_texts, max_features)

    X_train = bow_transform(train_texts, vocab)
    X_test = bow_transform(test_texts, vocab)

    return vocab, X_train, X_test


# ------------------------
# TF-IDF
# ------------------------

def compute_idf(train_texts, vocab):
    """
    Computes IDF values.
    """

    N = len(train_texts)

    df = defaultdict(int)

    for text in train_texts:

        words = set(tokenize(text))

        for word in words:
            if word in vocab:
                df[word] += 1

    idf = np.zeros(len(vocab))

    for word, index in vocab.items():
        idf[index] = np.log(N / (df[word] + 1))

    return idf


def tfidf_transform(texts, vocab, idf):
    """
    Computes TF-IDF matrix.
    """

    X = np.zeros((len(texts), len(vocab)))

    for i, text in enumerate(texts):

        words = tokenize(text)

        counts = Counter(words)

        total_words = len(words)

        for word, count in counts.items():
            if word in vocab:
                tf = count / total_words
                X[i, vocab[word]] = tf * idf[vocab[word]]

    return X


def get_tfidf_features(train_texts, test_texts, max_features=None):

    vocab = build_vocabulary(train_texts, max_features)

    idf = compute_idf(train_texts, vocab)

    X_train = tfidf_transform(train_texts, vocab, idf)
    X_test = tfidf_transform(test_texts, vocab, idf)

    return vocab, X_train, X_test
