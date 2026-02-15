# Sports vs Politics Text Classification

This project implements a text classification system that classifies news articles as either **Sports** or **Politics** using classical machine learning techniques. Feature representations including **Bag of Words (BoW)** and **TF-IDF** are implemented **from scratch** without using built-in vectorizers. Multiple machine learning models are trained and compared to evaluate performance.

---

## Dataset

The project uses the HuffPost News dataset containing news articles with fields such as headline, short description, and category. Only articles belonging to the **SPORTS** and **POLITICS** categories are used. The headline and short description are combined to form the input text, and the category is used as the label.

---

## Feature Engineering

Two feature representation methods are implemented from scratch:

- **Bag of Words (BoW):** Represents each document as a vector of word frequency counts.
- **TF-IDF (Term Frequency–Inverse Document Frequency):** Weighs words based on their importance across the dataset.

The implementation includes tokenization, vocabulary construction, and vector generation without relying on external feature extraction libraries.

---

## Models Used

The following machine learning models are trained and evaluated:

- Multinomial Naive Bayes  
- Logistic Regression  
- Support Vector Machine (Linear SVM)  

Each model is trained using both BoW and TF-IDF features, and their performance is compared using accuracy, precision, recall, and F1-score.

---

## How to Run

Install dependencies and run the main script:
1:pip install -r requirements.txt
2:python main.py
3:that's it
