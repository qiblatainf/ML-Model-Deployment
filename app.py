from flask import Flask, jsonify, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('iris_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json()

    # Extract the features from the data
    features = [data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]
    features = [float(f) for f in features]

    # Perform prediction using the loaded model
    prediction = model.predict([features])
    predicted_class = int(prediction[0])

    # Return the prediction as JSON response
    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
