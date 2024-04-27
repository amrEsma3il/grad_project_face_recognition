import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from flask import Flask, request

# Load the trained model
model = load_model('face_recognition_model.h5')

# Function to create a label map with the names of subdirectories in the given directory
def create_label_map(directory_path):
    label_map = {}  # Mapping from label (person) to directory name
    
    # Get the list of subdirectories in the given directory
    subdirectories = [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]
    
    # Assign each subdirectory name to a label
    for label_id, directory_name in enumerate(subdirectories):
        label_map[label_id] = directory_name
            
    return label_map

# Define the directory containing the training data
train_dir = 'images/train'

# Create the label map
label_map = create_label_map(train_dir)

# Define the confidence threshold for known persons
known_threshold = 0.7  # Adjust this threshold as needed

# Function to recognize faces in an image
def recognize_faces(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Preprocess the image (resize and normalize)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Perform prediction
    predictions = model.predict(image)
    
    # Get the predicted label (person) and confidence
    predicted_label = np.argmax(predictions)
    confidence = np.max(predictions)
    
    # Get the person's name from the label map
    person_name = label_map.get(predicted_label, 'Unknown')
    
    # Check if the confidence is above the known threshold
    if confidence >= known_threshold:
        return {"predict":person_name}
    else:
        return {"predict":"unknown"}



app = Flask(__name__)

@app.route('/predict', methods=["POST","GET"])
def predict():
   return recognize_faces("1.jpg")

if __name__ == "__main__":
    app.run(debug=True)
