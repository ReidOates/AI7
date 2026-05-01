from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

app = Flask(__name__)

# Load model, scaler, and columns
try:
    model = tf.keras.models.load_model('model.keras')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('columns.pkl', 'rb') as f:
        columns = pickle.load(f)
except Exception as e:
    print(f"Error loading model assets: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Prepare data for processing
        input_data = {
            'year': [float(data['year'])],
            'age': [float(data['age'])],
            'hypertension': [int(data['hypertension'])],
            'heart_disease': [int(data['heart_disease'])],
            'bmi': [float(data['bmi'])],
            'hbA1c_level': [float(data['hbA1c_level'])],
            'blood_glucose_level': [float(data['blood_glucose_level'])]
        }
        
        # Gender encoding (Female is baseline)
        gender = data['gender']
        input_data['gender_Male'] = [1 if gender == 'Male' else 0]
        input_data['gender_Other'] = [1 if gender == 'Other' else 0]
        
        # Smoking history encoding (No Info is baseline)
        smoking = data['smoking_history']
        input_data['smoking_history_current'] = [1 if smoking == 'current' else 0]
        input_data['smoking_history_ever'] = [1 if smoking == 'ever' else 0]
        input_data['smoking_history_former'] = [1 if smoking == 'former' else 0]
        input_data['smoking_history_never'] = [1 if smoking == 'never' else 0]
        input_data['smoking_history_not current'] = [1 if smoking == 'not current' else 0]
        
        # Create DataFrame and ensure column order
        df_input = pd.DataFrame(input_data)
        df_input = df_input[columns]
        
        # Scale input
        scaled_input = scaler.transform(df_input)
        
        # Predict
        prediction_prob = model.predict(scaled_input)[0][0]
        prediction_label = 1 if prediction_prob > 0.5 else 0
        
        # Human readable result
        result = {
            'probability': float(prediction_prob),
            'prediction': int(prediction_label),
            'risk': 'Tinggi' if prediction_label == 1 else 'Rendah',
            'confidence': f"{prediction_prob * 100:.2f}%" if prediction_label == 1 else f"{(1 - prediction_prob) * 100:.2f}%"
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
