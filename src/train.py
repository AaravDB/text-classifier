# src/train.py

"""
Handles model training only.
"""


def train_model(model, X_train, y_train):
    """
    Trains the given model.

    Parameters:
        model: sklearn model instance
        X_train: training features
        y_train: training labels

    Returns:
        trained model
    """

    print(f"Training {model.__class__.__name__}...")

    model.fit(X_train, y_train)

    print("Training complete.")

    return model
