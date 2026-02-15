"""
Defines all machine learning models used in the project.
Each function returns an untrained model instance.
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def get_naive_bayes():
    """
    Multinomial Naive Bayes model.
    Best suited for text classification.
    """
    return MultinomialNB()


def get_logistic_regression():
    """
    Logistic Regression model.

    class_weight='balanced' handles class imbalance.
    """
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )


def get_svm():
    """
    Linear Support Vector Machine.

    LinearSVC is efficient for high-dimensional text data.
    """
    return LinearSVC(
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    )
