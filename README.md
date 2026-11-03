# SMS Spam Detection using Machine Learning

## Overview

This project focuses on classifying SMS messages as **Spam** or **Ham (Legitimate)** using Natural Language Processing (NLP) and machine learning techniques.

The objective is to build and compare classification models that can identify spam messages while minimizing false positives and missed spam messages.

Two machine learning algorithms were implemented and evaluated:

- **Multinomial Naive Bayes**
- **Support Vector Machine (SVM)** with different kernel functions

The project also explores text preprocessing, TF-IDF vectorization, dimensionality reduction using Truncated SVD, and hyperparameter experimentation.

## Dataset

**Dataset:** SMS Spam Collection Dataset

The dataset contains SMS messages labelled as either ham or spam.

| Attribute | Description |
|---|---|
| `label` | Message category: ham or spam |
| `message` | Original SMS message |

### Dataset preparation

- Original dataset: 5,572 messages.
- Removed unnecessary unnamed columns.
- Renamed columns to `label` and `message`.
- Removed 403 duplicate messages.
- Final dataset: **5,169 messages**.

### Class distribution

The dataset is imbalanced, with legitimate messages significantly outnumbering spam messages.

| Class | Approximate distribution |
|---|---:|
| Ham | 86.59% |
| Spam | 13.41% |

This imbalance makes accuracy alone insufficient for evaluating model performance. Precision, recall, and F1-score are also considered.

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- Jupyter Notebook / Google Colab

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

The dataset was explored to understand:

- Distribution of spam and ham messages.
- Class imbalance.
- Duplicate records.
- Message characteristics and text patterns.

### 2. Text Preprocessing

A preprocessing pipeline was applied to prepare SMS messages for model training.

Operations included:

- Converting text to lowercase.
- Removing punctuation and special characters.
- Tokenizing messages into individual words.
- Removing English stopwords.

Negation words such as `not` and `no` were retained because they can change the meaning of a message.

### 3. TF-IDF Vectorization

Text data cannot be directly processed by traditional machine learning algorithms. Therefore, TF-IDF (Term Frequency–Inverse Document Frequency) was used to convert messages into numerical feature vectors.

TF-IDF assigns importance to words based on their frequency in a message and their distribution across the document collection.

The vectorizer was fitted on training data and subsequently used to transform validation and test data to prevent data leakage.

### 4. Model 1: Multinomial Naive Bayes

Multinomial Naive Bayes was implemented as the baseline classification model.

It is commonly used for text classification because it works effectively with word-frequency-based representations.

**Evaluation metrics:**

| Metric | Score |
|---|---:|
| Accuracy | 96.03% |
| Precision | 98.91% |
| Recall | 69.47% |
| F1-score | 81.61% |

Naive Bayes achieved high precision but comparatively lower recall, indicating that it missed a considerable number of actual spam messages.

### 5. Model 2: Support Vector Machine (SVM)

Support Vector Machine was implemented to explore whether a more flexible classification approach could improve spam detection performance.

Three kernel functions were evaluated:

- Linear
- Polynomial
- Radial Basis Function (RBF)

The experiments included multiple values of the regularization parameter `C`:

`[1, 10, 50, 100, 200, 500, 1000]`

Class weights were balanced to account for the imbalanced dataset.

### 6. Dimensionality Reduction using Truncated SVD

The TF-IDF representation initially contained thousands of features.

Truncated Singular Value Decomposition (SVD) was applied to reduce the dimensionality of the sparse TF-IDF feature matrix.

The following component counts were investigated:

`[50, 100, 200, 300, 500]`

The reduced representations were used to train and validate SVM models.

This resulted in **105 SVM experiments** across different component counts, kernels, and regularization values.

The best-performing configuration was:

| Hyperparameter | Selected value |
|---|---|
| SVD components | 500 |
| Kernel | RBF |
| C | 1 |
| Class weight | Balanced |

The final SVM was retrained using the complete training dataset before evaluation on the held-out test set.

## Model Evaluation

Both models were evaluated on the same untouched test set containing 1,034 messages.

### Performance comparison

| Metric | Naive Bayes | SVM |
|---|---:|---:|
| Accuracy | 96.03% | **98.16%** |
| Precision | **98.91%** | 94.44% |
| Recall | 69.47% | **90.84%** |
| F1-score | 81.61% | **92.61%** |

### SVM Confusion Matrix

| | Predicted Ham | Predicted Spam |
|---|---:|---:|
| Actual Ham | 896 | 7 |
| Actual Spam | 12 | 119 |

The SVM correctly identified 119 out of 131 spam messages while incorrectly classifying 7 legitimate messages as spam.

### Key Findings

- SVM achieved higher accuracy and F1-score than Naive Bayes.
- Spam recall improved from 69.47% to 90.84%.
- SVM reduced missed spam messages from 40 to 12.
- Naive Bayes achieved higher precision, resulting in fewer legitimate messages being incorrectly flagged as spam.
- The RBF kernel with 500 SVD components and C = 1 provided the best validation F1-score among the configurations tested.

**Final Model:** The RBF-based Support Vector Machine was selected as the stronger overall model because of its improved spam detection recall and F1-score.


## Future Improvements

- Experiment with Logistic Regression and additional classification algorithms.
- Explore alternative text representations such as Word2Vec.
- Investigate further hyperparameter optimization.
- Evaluate performance on a larger and more diverse SMS dataset.
- Develop a simple interface for classifying user-provided messages.

## Conclusion

This project demonstrates an end-to-end text classification workflow, from exploratory data analysis and text preprocessing to feature extraction, dimensionality reduction, model experimentation, and evaluation.

The comparison between Multinomial Naive Bayes and Support Vector Machine highlights the importance of evaluating models using multiple metrics, particularly when working with imbalanced datasets.

The final SVM model achieved **98.16% accuracy and a 92.61% F1-score** on the held-out test set.

---
