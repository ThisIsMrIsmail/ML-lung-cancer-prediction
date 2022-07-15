import joblib
import numpy as np
from flask import Flask, request, render_template

# Create flask app
flask_app = Flask(__name__)
model = joblib.load('RF.pkl', 'rb')


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    if prediction == 1:
        result = "The person has lung cancer disease"
    elif prediction == 0:
        result = "The person does not has lung cancer disease"
    else:
        result = "Error"
    return render_template("index.html", prediction_text="{}".format(result))


if __name__ == "__main__":
    flask_app.run(debug=True)
