from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)
model = joblib.load(os.path.join("model", "model.pkl"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
