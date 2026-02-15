# src/save_load.py

import pickle
import os




def save_model(model, filename):
    """
    Saves trained model to saved_models folder.
    """

    os.makedirs("saved_models", exist_ok=True)

    filepath = os.path.join("saved_models", filename)

    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def load_model(filename):
    """
    Loads saved model.
    """

    filepath = os.path.join("saved_models", filename)

    with open(filepath, "rb") as f:
        model = pickle.load(f)

    return model
