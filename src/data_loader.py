# src/data_loader.py

import pandas as pd


def load_data(filepath):
    """
    Loads HuffPost dataset, filters SPORTS and POLITICS,
    combines text fields, and converts labels to numeric.

    Parameters:
        filepath (str): path to dataset json file

    Returns:
        texts (list): list of input text
        labels (list): list of numeric labels (0 = Sports, 1 = Politics)
    """

    # Load JSON dataset (HuffPost format uses lines=True)
    df = pd.read_json(filepath, lines=True)

    print("Original dataset size:", len(df))

    # Keep only SPORTS and POLITICS
    df = df[df['category'].isin(['SPORTS', 'POLITICS'])]

    print("Filtered dataset size:", len(df))
    print("\nClass distribution:")
    print(df['category'].value_counts())

    # Combine headline and short_description
    df['text'] = df['headline'] + " " + df['short_description']

    # Convert labels to numeric
    df['label'] = df['category'].map({
        'SPORTS': 0,
        'POLITICS': 1
    })

    # Remove missing values if any
    df = df.dropna(subset=['text', 'label'])

    # Return lists
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    return texts, labels
