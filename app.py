from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("dropout_model1.pkl")
cluster_model = joblib.load("cluster_model1.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        features = np.array([data])

        pred = model.predict(features)
        clu = cluster_model.predict(features)

        result = "At Risk of Dropout" if pred[0] == 1 else "Safe Student"
        cluster_result = f"Cluster Group: {clu[0]}"

        return render_template('index.html',
                               prediction_text=result,
                               cluster_text=cluster_result)

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
