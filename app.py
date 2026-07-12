from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# -------------------------
# Train Machine Learning Model
# -------------------------
days = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
temps = np.array([30,31,30,29,31,32,31,30,29,30])

model = LinearRegression()
model.fit(days, temps)

# -------------------------
# Weather Condition
# -------------------------
def weather_condition(temp):
    if temp >= 32:
        return "☀️ Sunny"
    elif temp >= 30:
        return "⛅ Cloudy"
    elif temp >= 28:
        return "🌤 Pleasant"
    else:
        return "🌧 Rainy"

# -------------------------
# Home
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------
# Prediction
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():

    day = int(request.form["day"])

    prediction = model.predict([[day]])[0]

    condition = weather_condition(prediction)

    return render_template(
        "index.html",
        prediction=round(prediction,2),
        condition=condition,
        day=day
    )

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)