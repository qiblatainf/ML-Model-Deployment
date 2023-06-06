# Import necessary libraries
from flask import Flask, request, jsonify
import joblib

# Load the trained machine learning model
model = joblib.load('your_model_file.pkl')

# Create Flask app
app = Flask(__name__)

# Define a route for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json()

    # Perform prediction using the loaded model
    prediction = model.predict(data)

    # Return the prediction as JSON response
    return jsonify({'prediction': prediction.tolist()})

# Run the Flask app
if __name__ == '__main__':
    app.run()
