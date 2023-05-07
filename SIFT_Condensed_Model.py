import os

import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm

# Set the image dimensions and SIFT parameters
IMG_WIDTH = 224
IMG_HEIGHT = 224
SIFT_FEATURES = 128

# Function to extract SIFT features from images
def extract_sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    _, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is not None:
        if descriptors.shape[0] < SIFT_FEATURES:
            descriptors = np.pad(descriptors, ((0, SIFT_FEATURES - descriptors.shape[0]), (0, 0)), mode='constant')
        descriptors = descriptors[:SIFT_FEATURES]
        return descriptors.flatten()
    else:
        return np.zeros(SIFT_FEATURES)

# Load and preprocess the images, and extract SIFT features
def load_images(directory):
    images = []
    labels = []

    for label in os.listdir(directory):
        label_directory = os.path.join(directory, label)
        for image_file in tqdm(os.listdir(label_directory), desc='Processing ' + label):
            image_path = os.path.join(label_directory, image_file)

            image = cv2.imread(image_path)
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

            sift_features = extract_sift_features(image)

            images.append(sift_features)
            labels.append(label)

    return images, labels

# Load and preprocess the images
directory = "compressed_dataset"
images, labels = load_images(directory)
images = np.array(images)
labels = np.array(labels)

# Encode the location labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Create an MLP model
mlp = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', random_state=42)

# Train the MLP model
train_losses = []
train_accuracies = []
epochs = 100

for epoch in range(epochs):
    mlp.partial_fit(X_train, y_train)
    train_predictions = mlp.predict(X_train)
    train_loss = mlp.loss_
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

# Make predictions on the testing data
test_predictions = mlp.predict(X_test)

# Calculate the accuracy of the model on testing data
test_accuracy = accuracy_score(y_test, test_predictions)
print("Testing Accuracy:", test_accuracy)

# Plot the training loss and accuracy
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_losses)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(range(epochs), train_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy')

plt.tight_layout()
plt.show()

# Save the trained MLP model to a file
model_filename = 'trained_model.pkl'
joblib.dump(mlp, model_filename)
print("Model saved")