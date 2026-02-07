import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

df = pd.read_csv("spam.csv", encoding="latin-1")

if 'v1' in df.columns and 'v2' in df.columns:
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
elif 'label' in df.columns and 'text' in df.columns:
    df = df[['label', 'text']]
    df.columns = ['label', 'message']
elif 'category' in df.columns and 'message' in df.columns:
    df = df[['category', 'message']]
    df.columns = ['label', 'message']
else:
    raise ValueError(f"Unknown column format: {df.columns}")


df['label'] = df['label'].map({'ham': 0, 'spam': 1})


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df['clean_message'] = df['message'].apply(clean_text)

vectorizer = TfidfVectorizer(
    max_features=3000,
    stop_words='english'
)

X = vectorizer.fit_transform(df['clean_message'])
y = df['label']

model = SVC(kernel='linear')
model.fit(X, y)


pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully")
