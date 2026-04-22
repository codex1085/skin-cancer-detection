from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import io
from tensorflow.keras.preprocessing import image

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model
model = load_model('/content/drive/MyDrive/SkinCancerDetection/models/skin_cancer_model.keras')

# Set the image size for prediction
IMAGE_SIZE = (224, 224)

# Define the classes
class_names = ['benign', 'malignant']  # Update this based on your model output classes

# Helper function for image preprocessing
def preprocess_image(img):
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Home route (index.html)
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route (handles image upload and prediction)
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        
        # If no file is selected
        if file.filename == '':
            return redirect(request.url)
        
        # Read the image
        img = Image.open(file.stream)
        img_array = preprocess_image(img)
        
        # Predict the class
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        
        return render_template('index.html', prediction_text=f"Predicted Class: {predicted_class}")

if __name__ == "__main__":
    app.run(debug=True)
