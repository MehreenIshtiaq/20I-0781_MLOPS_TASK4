from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('california_housing_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the features from the POST request
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    
    # Use the model to make a prediction
    prediction = model.predict(features)
    
    # Send the prediction back as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
