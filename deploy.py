from flask import Flask, request, jsonify, render_template
import pickle
import csv
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
import io

app = Flask(__name__)
rf_model = pickle.load(open('random_forest_model.pkl', 'rb'))
# imputer = SimpleImputer(strategy='mean')


@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')

def predict(data):
    imputer = SimpleImputer(strategy='mean')
    data = imputer.fit_transform(data)
    y_pred = rf_model.predict(data)
    return y_pred

        
        
@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.json['data']
    data = [list(data.values())]
    data = np.array(data, dtype=float)  # Convert data to a NumPy array
    pred = predict(data)
    return jsonify(pred)

if __name__ == '__main__':
    app.run(debug=True)
