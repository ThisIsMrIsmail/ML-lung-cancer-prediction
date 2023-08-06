import joblib
import numpy as np
import flask as flask

app = flask.Flask(__name__)
model = joblib.load('model.pkl', 'rb')


@app.route("/")
def home():
    return flask.render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in flask.request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    
    if prediction == 1:
        result = "Suspicion to have the Disease"
    elif prediction == 0:
        result = "Normal Person"
    else:
        result = "Error! try again"
    return flask.render_template("index.html", prediction_text="{}".format(result))


if __name__ == "__main__":
    app.run(host='0.0.0.0')