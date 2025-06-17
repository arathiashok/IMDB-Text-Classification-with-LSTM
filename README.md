# IMDB Sentiment Classification with LSTM

This project implements a binary sentiment classifier using an LSTM-based neural network in PyTorch. It classifies IMDB movie reviews as positive or negative based on their textual content.

---

## 🎯 Objective

- Build an end-to-end pipeline for sentiment analysis on the IMDB dataset using deep learning.
- Explore text preprocessing, LSTM model architecture, and evaluation metrics.
- Tune hyperparameters (hidden size, learning rate, epochs) and analyze performance.

---

## 📚 Dataset

- **Source:** [IMDB Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
- **Classes:** Positive (1), Negative (0)
- **Size:** 25,000 training + 25,000 testing reviews

---

## 🏗️ Model Architecture

- **Embedding Layer**
- **LSTM Layer:** Tuned for hidden sizes (64, 128)
- **Fully Connected Layer**
- **Sigmoid Output**

---

## 🔍 Preprocessing Steps

- Lowercasing  
- Punctuation removal  
- Tokenization  
- Stopword removal  
- Stemming  

---

## 🧪 Experimental Results

| Metric      | Score     |
|-------------|-----------|
| Accuracy    | 79.9%     |
| Precision   | 75.5%     |
| Recall      | 88.7%     |
| F1 Score    | 81.6%     |
| AUC         | 0.8721    |

---

## 🧠 Key Findings

- Best validation accuracy (81.2%) achieved with LSTM hidden size = 128.
- Preprocessing significantly reduced noise, improving model performance.
- Balanced validation set ensured generalizable results (1.5K pos & 1.5K neg samples).
- AUC metric highlighted strong model ability to distinguish sentiment under imbalanced settings.

---

## 🛠️ Setup

```bash
pip install torch nltk scikit-learn matplotlib
