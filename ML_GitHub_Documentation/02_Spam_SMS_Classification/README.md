# Spam SMS Classification

A natural-language processing project that classifies SMS messages as **Spam** or **Ham** using TF-IDF and Multinomial Naive Bayes.

## Overview
- Dataset: SMS Spam Collection
- Original rows: 5,572
- Duplicate rows removed: 403
- Final rows: 5,169
- Model: Multinomial Naive Bayes
- Text representation: TF-IDF
- Recorded accuracy: **97.78%**

## Workflow
1. Load and inspect the dataset
2. Check missing values and duplicates
3. Remove duplicate messages
4. Explore class distribution
5. Create message-length and word-count features
6. Convert text to numerical vectors with TF-IDF
7. Train Multinomial Naive Bayes
8. Evaluate with classification metrics
9. Test custom messages
10. Save model and vectorizer with Joblib

## Important Concept
TF-IDF is the **feature extraction method**; Multinomial Naive Bayes is the **classifier**.

The vectorizer is fitted on training text with `fit_transform()` and applied to test/new text with `transform()`.

## Saved Artifacts
```text
spam_classifier.pkl
tfidf_vectorizer.pkl
```

Both are needed for inference because new messages must be transformed using the same fitted TF-IDF representation.

## Technologies
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, WordCloud, Joblib

## Structure
```text
02_Spam_SMS_Classification/
├── spam_sms_classification.ipynb
├── README.md
└── DOCUMENTATION.md
```

## Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud joblib
jupyter notebook
```

## Limitations & Future Work
This is a portfolio implementation, not a production spam-filtering service. Future improvements could include n-grams, stronger preprocessing, cross-validation, false-positive analysis, model comparison, monitoring, and deployment.

## Author
**Aditya Roy**
