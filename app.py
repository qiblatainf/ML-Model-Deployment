from flask import Flask, request, jsonify
import joblib

# Load the saved model
model = joblib.load('iris_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json()

    # Convert the data into a format compatible with the model
    features = [data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]
    features = [float(f) for f in features]

    # Perform prediction using the loaded model
    prediction = model.predict([features])
    predicted_class = int(prediction[0])

    # Return the prediction as JSON response
    species = data.target_names[predicted_class]
    return jsonify({'species': species})

if __name__ == '__main__':
    app.run()
