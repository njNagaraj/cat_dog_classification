from flask import Flask, request, jsonify, render_template
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import joblib

app = Flask(__name__)

# Load the trained model
MODEL_FILE = 'svm_model.pkl'
model = joblib.load(MODEL_FILE)

Categories = ['cats', 'dogs']

@app.route('/classify_image', methods=['POST'])
def classify_image():
    try:
        # Receive the image file from the request
        image_file = request.files.get('image')
        if not image_file:
            return jsonify({'error': 'No image file provided'}), 400

        # Save the image to a temporary location
        temp_path = 'temp.jpg'
        image_file.save(temp_path)

        # Load and preprocess the image
        img_array = imread(temp_path)
        img_resized = resize(img_array, (50, 50, 3))
        img_flattened = [img_resized.flatten()]

        # Predict the class probabilities
        probabilities = model.predict_proba(img_flattened)

        predicted_class = Categories[model.predict(img_flattened)[0]]

        # Format the response
        response_data = {
            'predicted_class': predicted_class,
        }

        # Delete the temporary image file
        os.remove(temp_path)

        # Return the result to the Flutter application
        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
