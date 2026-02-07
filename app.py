from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None   # 👈 IMPORTANT

    if request.method == "POST":
        email = request.form.get("email", "")
        cleaned = clean_text(email)
        vector = vectorizer.transform([cleaned])
        result = model.predict(vector)[0]
        prediction = "✅ Not Spam" if result == 0 else "🚨 Spam"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
