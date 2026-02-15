from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, X_test, y_test):
    """
    Evaluates trained model and returns metrics.
    """

    y_pred = model.predict(X_test)

    results = {

        "accuracy": accuracy_score(y_test, y_pred),

        "precision": precision_score(y_test, y_pred),

        "recall": recall_score(y_test, y_pred),

        "f1_score": f1_score(y_test, y_pred)
    }

    return results
