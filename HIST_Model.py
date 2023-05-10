import cv2
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

# Set the image dimensions
IMG_WIDTH = 512
IMG_HEIGHT = 220

random_state = 22


# Function to extract color histogram features from images
def extract_color_histogram(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Compute the color histogram
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [32, 32, 32], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    return hist


# Load and preprocess the images, and extract color histogram features
def load_images(directory):
    images = []
    labels = []

    for label in os.listdir(directory):
        label_directory = os.path.join(directory, label)
        for image_file in os.listdir(label_directory):
            image_path = os.path.join(label_directory, image_file)

            # Load the image
            image = cv2.imread(image_path)
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

            # Extract the color histogram features
            hist = extract_color_histogram(image)

            # Append the features and label to the lists
            images.append(hist)
            labels.append(label)

    return images, labels


# Load and preprocess the images
directory = "compressed_dataset"
images, labels = load_images(directory)

# Convert the image data and labels to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Encode the location labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=random_state)

# Create an MLP model
mlp = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', random_state=random_state)

# Train the MLP model
train_losses = []
train_accuracies = []
epochs = 100  # Number of training epochs

for epoch in range(epochs):
    mlp.partial_fit(X_train, y_train, classes=np.unique(y_train))
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

model_filename = 'trained_model.pkl'
joblib.dump(mlp, model_filename)
print("Model saved as", model_filename)

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
