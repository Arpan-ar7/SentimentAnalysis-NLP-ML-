from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(
    open("sentiment_model.pkl", "rb")
)

# Load TF-IDF
tfidf = pickle.load(
    open("tfidf.pkl", "rb")
)

# Emotion mapping
labels = {
    0: "Sadness 😢",
    1: "Anger 😡",
    2: "Love ❤️",
    3: "Surprise 😲",
    4: "Fear 😨",
    5: "Joy 😄"
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    text = request.form["text"]

    # TF-IDF transform
    vector = tfidf.transform([text])

    # Prediction
    pred = model.predict(vector)[0]

    # Probability (optional)
    prob = model.predict_proba(vector)

    emotion = labels[pred]

    confidence = round(
        max(prob[0]) * 100,
        2
    )

    return render_template(
        "index.html",
        prediction=emotion,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)