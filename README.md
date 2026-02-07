# Email Spam Classifier using SVM

This project is an end-to-end **Email Spam Detection system** built using **Support Vector Machine (SVM)** and **TF-IDF vectorization**.  
The model classifies emails as **Spam** or **Not Spam (Ham)** based on their textual content.

---

## Project Overview

Spam emails are a major problem in digital communication.  
This project applies **Machine Learning and Natural Language Processing (NLP)** techniques to automatically identify spam emails with high accuracy.

---

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Support Vector Machine(SVM)
- GridSearchCV 

---

## Dataset

- Dataset: SMS Spam Collection Dataset
- Source: Kaggle / UCI ML Repository
- Labels:
  - `spam` → 1
  - `ham` → 0

---

## Project Workflow

1. Load and inspect dataset
2. Text preprocessing (cleaning, normalization)
3. Feature extraction using TF-IDF
4. Train-test split
5. Train SVM with linear kernel
6. Hyperparameter tuning using GridSearchCV
7. Model evaluation
8. Test on custom email input

---

## Model Details

- Algorithm: Support Vector Machine (SVM)
- Kernel: Linear
- Reason: Text data is high-dimensional and sparse
- Optimization: GridSearchCV for tuning parameter `C`

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- Confusion Matrix

---

## Sample Prediction

```python
predict_spam("Congratulations! You won a free prize")
# Output: Spam

predict_spam("Are we meeting at 5 pm today?")
# Output: Not Spam


## Project Structure

spam-svm-project/
│
├── spam.csv
├── spam_classifier.ipynb
├── README.md
└── requirements.txt
---
## Author
Raghu Ram
BTech Student | Aspiring AI Engineer
