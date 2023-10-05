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

# @app.route("/upload", methods=["POST"])
# # def upload():
# #     if request.method == "POST":
# #         file = request.files["file"]
# #         reader = csv.reader(file)
# #         data = list(reader)
# #         data = np.array(data, dtype=float)  # Convert data to a NumPy array
# #         pred = predict(data)
# #         return render_template('index.html', prediction_result=pred)

# @app.route("/upload", methods=["POST"])
# def upload():
#     if request.method == "POST":
#         file = request.files["file"]
#         data = []
        
#         # Ensure the file is opened in text mode ('rt' for read text)
#         with file.stream as csvfile:
#             reader = csv.reader(csvfile)
#             for row in reader:
#                 data.append(row)
        
#         pred = predict(data)
#         return render_template('index.html', prediction_result=pred)

# def upload():
#     if request.method == "POST":
#         file = request.files["file"]
#         data = []
        
#         # Ensure the file is opened in text mode ('r' for read mode)
#         with io.TextIOWrapper(file.stream, encoding='utf-8') as csvfile:
#             reader = csv.reader(csvfile)
#             for row in reader:
#                 data.append(row)
        
#         pred = predict(data)
#         return render_template('index.html', prediction_result=pred)

# @app.route("/upload", methods=["POST"])
# def upload():
#     if request.method == "POST":
#         file = request.files["file"]
#         data = []
        
#         # Read the contents of the file using the 'read' method
#         file_contents = file.read().decode('utf-8')
        
#         # Use 'io.StringIO' to treat the contents as a text stream
#         csv_stream = io.StringIO(file_contents)
        
#         # Now, you can read the CSV data from the stream
#         reader = csv.reader(csv_stream)
#         for row in reader:
#             data.append(row)
        
#         pred = predict(data)
#         return render_template('index.html', prediction_result=pred)

# def upload():
#     if request.method == "POST":
#         file = request.files["file"]
#         if file:
#             # Read the uploaded CSV file into a pandas DataFrame
#             df = pd.read_csv(file)
#             # Assuming 'cls' is the target column in your model
#             X = df.drop("cls", axis=1)
#             data = X.values  # Extract the data as a NumPy array
#             pred = predict(data)
#             return render_template('index.html', prediction_result=pred)
#         else:
#             return render_template('index.html', error="No file selected")


# def upload():
#     if request.method == "POST":
#         file = request.files["file"]
#         if file:
#             # Read the uploaded CSV file into a pandas DataFrame
#             df = pd.read_csv(file)
            
#             # Assuming 'cls' is the target column in your model
#             X = df.drop("cls", axis=1)
            
#             # Preprocess the uploaded data (impute missing values, scale, etc.)
#             X_imputed = imputer.transform(X)
            
#             # Make predictions using the fitted model
#             pred = rf_model.predict(X_imputed)
            
#             return render_template('index.html', prediction_result=pred)
#         else:
#             return render_template('index.html', error="No file selected")
        
        
@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.json['data']
    data = [list(data.values())]
    data = np.array(data, dtype=float)  # Convert data to a NumPy array
    pred = predict(data)
    return jsonify(pred)

if __name__ == '__main__':
    app.run(debug=True)
