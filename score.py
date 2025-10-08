# score.py
import os
import joblib
import pandas as pd
import json

# init() runs once when the service starts.
def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created by Azure.
    # It points to the folder where the model file is downloaded.
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'churn_rf.joblib')
    model = joblib.load(model_path)
    print("Model loaded successfully.")

# run() runs for each API call.
def run(raw_data):
    try:
        # The API request body is passed as a JSON string.
        data_dict = json.loads(raw_data)['data']
        # The model expects a DataFrame, so we convert the dictionary.
        df = pd.DataFrame([data_dict])
        
        # Use the loaded model to make a prediction and get probabilities.
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0, 1]
        
        # Format the response as a dictionary.
        result = {'prediction': int(pred), 'probability': float(proba)}
        return result
    except Exception as e:
        error = str(e)
        return {'error': error}