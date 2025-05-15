from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model and scaler
model = joblib.load("expense_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Expense Predictor API. Use POST /predict to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    month = data['month']
    year = data['year']
    income = data['income']
    festival_count = data['festival_count']

    input_data = np.array([[month, year, income, festival_count]])
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)[0]

    return jsonify({
        "predictions": dict(zip([
            'Food', 'Groceries', 'Transport', 'Entertainment', 
            'Shopping', 'Rent', 'Bills', 'Healthcare', 'Education'
        ], [round(val, 2) for val in prediction]))
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
