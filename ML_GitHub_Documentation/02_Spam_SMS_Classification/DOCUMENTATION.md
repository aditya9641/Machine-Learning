# Spam SMS Classification — Documentation

## Objective
Build a binary classifier that distinguishes legitimate SMS (`ham`) from spam.

## Data Cleaning
The notebook loads `spam.csv`, keeps the first two relevant columns, renames them to `label` and `message`, checks missing values and duplicates, and removes 403 duplicate rows.

Before duplicate removal:
- Ham: 4,825
- Spam: 747

## Exploratory Analysis
The notebook examines class distribution, message length, word count, and common words using count plots and WordCloud.

Additional features:
```python
df["length"] = df["message"].apply(len)
df["word_count"] = df["message"].apply(lambda x: len(x.split()))
```

## TF-IDF
Raw text cannot be directly passed to the classifier. TF-IDF converts messages into numerical feature vectors by weighting terms according to their importance within the corpus.

## Model
The classifier is `MultinomialNB()`, a model commonly used for text classification with count/TF-IDF-style representations.

## Leakage Prevention
The intended transformation flow is:
```python
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
```
The vectorizer learns from training data and then transforms unseen data.

## Results
Recorded accuracy: **97.78%**. Precision, recall, and F1 should also be considered because accuracy alone can be misleading when classes are imbalanced.

## Persistence
```python
joblib.dump(model, "spam_classifier.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
```

## Inference
```text
New SMS → fitted TF-IDF vectorizer → numerical vector → Naive Bayes → Spam/Ham
```

## Implementation Accuracy
The original notebook uses TF-IDF + Multinomial Naive Bayes. It does not use transformers or deep-learning language models.

## Production Improvements
Add stronger preprocessing, cross-validation, threshold analysis, monitoring, retraining, and API deployment.
