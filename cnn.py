import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the paths to your dataset
train_dir = 'images/train'
test_dir = 'images/test'

# Function to load images and labels from a directory
def load_data_from_directory(directory):
    images = []
    labels = []
    label_map = {}  # Mapping from label (person) to integer ID
    label_id = 0
    
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            label_map[label] = label_id
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                image = cv2.imread(image_path)
                if image is not None:
                    # Resize image to 64x64 and append to the list
                    image = cv2.resize(image, (64, 64))
                    images.append(image)
                    labels.append(label_id)
            label_id += 1
            
    return np.array(images), np.array(labels), label_map


# Load training data
X_train, Y_train, label_map = load_data_from_directory(train_dir)

# Load testing data
X_test, Y_test, _ = load_data_from_directory(test_dir)

# Normalize pixel values to range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert labels to one-hot encoding
num_classes = len(label_map)
Y_train = to_categorical(Y_train, num_classes=num_classes)
Y_test = to_categorical(Y_test, num_classes=num_classes)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# # Define the CNN model architecture with additional hidden layers
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),  # Additional Convolutional Layer
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(256, activation='relu'),  # Additional Dense Layer
#     Dropout(0.5),
#     Dense(128, activation='relu'),  # Additional Dense Layer
#     Dropout(0.5),
#     Dense(num_classes, activation='softmax')
# ])

# Define the CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callback to save the best model during training
checkpoint = ModelCheckpoint('face_recognition_model.h5', monitor='val_loss', save_best_only=True, mode='min')

# Train the model
model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[checkpoint])

# Save the trained model
model.save('face_recognition_model.h5')





from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report


# Load the trained model
model = load_model('face_recognition_model.h5')

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, Y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Predict labels for the test set
Y_pred = model.predict(X_test)
predicted_labels = np.argmax(Y_pred, axis=1)

# Convert one-hot encoded labels to integer labels
true_labels = np.argmax(Y_test, axis=1)

# Calculate accuracy manually
manual_accuracy = np.mean(predicted_labels == true_labels)
print("Manual Test Accuracy:", manual_accuracy)

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate classification report
class_report = classification_report(true_labels, predicted_labels)
print("Classification Report:")
print(class_report)