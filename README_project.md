# Fake News Detection with Machine Learning and Transformers

## Overview

This is for Ironhack, project 3 NLP. This project aims to classify news headlines as **real** or **fake** using Natural Language Processing (NLP). It combines classical machine learning models with modern pre-trained transformer-based models to detect misinformation effectively.

Project Overview
This project aims to build a classification model that distinguishes between real and fake news headlines. Using Natural Language Processing (NLP) techniques, we trained both traditional ML models and transformer-based models on labeled headline data, and evaluated their ability to generalize to unseen text.

---

## Dataset

We worked with two files:
- `training_data.csv`: Contains labeled headlines  
  - `0`: Fake News  
  - `1`: Real News  
- `testing_data.csv`: Contains unlabeled headlines used for final predictions after model training

---

## Preprocessing
We applied preprocessing depending on the model:

### Logistic Regression
- Converted text to lowercase
- Removed punctuation and numbers
- Tokenized (NLTK)
- Removed stopwords
- Lemmatized tokens using WordNetLemmatizer

### Random Forest:
- Used raw headlines
- Preprocessing handled within the TfidfVectorizer via:
    - stop_words='english'
    - ngram_range=(1,2)
    - max_features=5000

### Transformer Models:    
- Transformer models used **built-in tokenizers** (e.g., BERT’s tokenizer). Only headers were added, no manual text or custom preprocessing was applied   

### Vectorization
Used **TF-IDF Vectorizer** with:
- `max_features=5000` (LinearRegression, RandomForest)
- `ngram_range=(1,2)` (RandomForest)
- `stop_words='english'` (LinearRegression, RandomForest)

---

## Exploratory Data Analysis (EDA)

EDA helped us understand dataset balance and variability in headline length across classes. We explored the following before and after preprocssing/cleaning:

- Label distribution (`real` vs `fake`)
- Headline length distribution
    - Boxplots to visualize outliers and length variability
    - Histograms to visualize distribution

---

## Models Trained

### 1. **Logistic Regression**
- TF-IDF vectorization with full preprocessing
- `max_iter=1000`
- ~93% accuracy on validation set

### 2. **Random Forest Classifier**
- Raw headlines passed to TF-IDF with built-in preprocessing
- `n_estimators=200`, `n_jobs=-1`, `random_state=42`
- ~93% validation accuracy

### 3. **Transformers**
We explored two models via Hugging Face:

| Model Name | Hugging Face Link | Result |
|------------|-------------------|--------|
| `distilbert-base-uncased-finetuned-sst-2-english` | [Link](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) | Not suitable (sentiment model) |
| `mrm8488/bert-tiny-finetuned-fake-news-detection` | [Link](https://huggingface.co/mrm8488/bert-tiny-finetuned-fake-news-detection) | Weak generalization |

---

## Evaluation Metrics (Validation Set) Linear Regression Model

- **Accuracy:** 0.93
- **Precision:**  
  - Fake (0): 0.94  
  - Real (1): 0.92  
- **Recall:**  
  - Fake (0): 0.92  
  - Real (1): 0.94  
- **F1-Score (both classes):** 0.93
- **Confusion Matrix** shows balanced classification with low false positives/negatives.

> **Accuracy Estimation**: Based on validation results, we estimate similar ~93% accuracy on the true test set.

---

## Test Set Prediction
- Ran final predictions using the best-performing model (Linear Regression), as well as close second (Random Forest).
- Output saved in the format required:  
  - **File:** `testing_data_predictions.csv`, `testing_data_predictions_RandomForest.csv`  
  - **Format:** No header, just `[Label, Headline]`

---

##  Reproducibility

1. Clone the repo
2. Import requirements.txt
3. Create a virtual environment (recommended):

```bash
conda create -n nlp_env python=3.9
conda activate nlp_env