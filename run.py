from flask import Flask, request, jsonify
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
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found.")

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
        img_flattened = img_resized.flatten()
        img_flattened = np.expand_dims(img_flattened, axis=0)

        # Predict the class probabilities
        probabilities = model.predict_proba(img_flattened)[0]
        # Get the predicted class
        predicted_class = Categories[np.argmax(probabilities)]
        # Get the probability of the predicted class
        confidence = probabilities[np.argmax(probabilities)]

        # Delete the temporary image file
        os.remove(temp_path)

        # Return the result to the Flutter application
        return jsonify({'predicted_class': predicted_class, 'confidence': confidence}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
