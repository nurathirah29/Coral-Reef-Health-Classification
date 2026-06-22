from flask import Flask, request, jsonify
import numpy as np
import cv2
import pickle

app = Flask(__name__)

# Load the serialized SVM model from the pickle file
with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Class labels mapping
dec = {0: 'Healthy', 1: 'Bleached'}
@app.route('/')
def index():
    return"Nur Athirah - Flask API"

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        # Handle GET request
        return "This endpoint only accepts POST requests. Use a POST request with an image to get predictions."
    
    elif request.method == 'POST':
        try:
            # Get the image data from the POST request
            image_data = request.files.get('image').read()

            # Preprocess the image (resize, normalize, reshape)
            img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (200, 200))
            img_normalized = img_resized / 255
            img_reshaped = img_normalized.reshape(1, -1)

            # Make predictions using the loaded SVM model
            prediction = model.predict(img_reshaped)
            predicted_class = dec[prediction[0]]

            # Predict the confidence (probability) of the image belonging to each class
            confidence = model.predict_proba(img_reshaped)[0]

            # Create a dictionary to hold the prediction and confidence
            result = {'Prediction': predicted_class, 'Confidence': {dec[0]: confidence[0], dec[1]: confidence[1]}}

            return jsonify(result)
        
        except Exception as e:
            return jsonify({'error': 'Error processing the request.', 'details': str(e)}), 500

    else:
        # Handle other HTTP methods (e.g., PUT, DELETE, etc.)
        return "Method Not Allowed"

if __name__ == '__main__':
    app.run(host='10.62.23.28', port=5000, debug=True)