import pandas as pd
import os
from src.save_load import save_model
"""
Runs all experiments:

Feature representations:
- Bag of Words (scratch)
- TF-IDF (scratch)

Models:
- Naive Bayes
- Logistic Regression
- SVM

"""

from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.features import get_bow_features, get_tfidf_features
from src.models import (
    get_naive_bayes,
    get_logistic_regression,
    get_svm
)
from src.train import train_model
from src.evaluate import evaluate_model


def run_all_experiments():

    print("Loading dataset...")

    texts, labels = load_data("data/news.json")


    # ------------------------
    # Train-test split
    # ------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )


    # ------------------------
    # Feature representations
    # ------------------------

    print("\nCreating Bag of Words features...")
    bow_vocab, X_train_bow, X_test_bow = get_bow_features(
        X_train, X_test, max_features=None
    )

    print("BoW vocabulary size:", len(bow_vocab))


    print("\nCreating TF-IDF features...")
    tfidf_vocab, X_train_tfidf, X_test_tfidf = get_tfidf_features(
        X_train, X_test, max_features=None
    )

    print("TF-IDF vocabulary size:", len(tfidf_vocab))


    # ------------------------
    # Models dictionary
    # ------------------------

    models = {

        "Naive Bayes": get_naive_bayes,

        "Logistic Regression": get_logistic_regression,

        "SVM": get_svm
    }

    # ------------------------
    # Feature sets dictionary
    # ------------------------

    feature_sets = {

        "Bag of Words": (X_train_bow, X_test_bow),

        "TF-IDF": (X_train_tfidf, X_test_tfidf)
    }


    # ------------------------
    # Run experiments
    # ------------------------

    results = []

    for feature_name, (X_train_feat, X_test_feat) in feature_sets.items():

        print(f"\n========== Feature: {feature_name} ==========")

        for model_name, model_function in models.items():

            print(f"\nRunning {model_name} with {feature_name}...")

            model = model_function()

            trained_model = train_model(model, X_train_feat, y_train)
            
            model_filename = f"{model_name}_{feature_name}.pkl"

            model_filename = model_filename.replace(" ", "_")

            save_model(trained_model, model_filename)

            print(f"Model saved as saved_models/{model_filename}")

            metrics = evaluate_model(trained_model, X_test_feat, y_test)

            result = {

                "Feature": feature_name,

                "Model": model_name,

                "Accuracy": metrics["accuracy"],

                "Precision": metrics["precision"],

                "Recall": metrics["recall"],

                "F1 Score": metrics["f1_score"]
            }

            results.append(result)

            print("Accuracy:", metrics["accuracy"])
            print("F1 Score:", metrics["f1_score"])


    # ------------------------
    # Print final summary
    # ------------------------

    print("\n\n========== FINAL RESULTS ==========")

    for r in results:

        print(
            f"{r['Model']} + {r['Feature']} → "
            f"Accuracy: {r['Accuracy']:.4f}, "
            f"F1: {r['F1 Score']:.4f}"
        )

  


        # Save results to CSV
    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame(results)

    df.to_csv("results/results.csv", index=False)

    print("\nResults saved to results/results.csv")
    return results


