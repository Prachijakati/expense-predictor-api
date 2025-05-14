from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load("expense_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.get_json()
    
    month = data['month']
    year = data['year']
    income = data['income']
    festival_count = data['festival_count']

    # Prepare input data for prediction
    input_data = np.array([[month, year, income, festival_count]])
    input_scaled = scaler.transform(input_data)
    
    # Predict the expenses
    prediction = model.predict(input_scaled)[0]

    # Return predictions as a JSON response
    return jsonify({
        "predictions": dict(zip([
            'Food', 'Groceries', 'Transport', 'Entertainment', 
            'Shopping', 'Rent', 'Bills', 'Healthcare', 'Education'
        ], [round(val, 2) for val in prediction]))
    })

import os 

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)