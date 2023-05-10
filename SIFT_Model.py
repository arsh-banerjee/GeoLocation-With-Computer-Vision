import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm

# Set the image dimensions and SIFT parameters
IMG_WIDTH = 512
IMG_HEIGHT = 220
SIFT_FEATURES = 512
NUM_CLUSTERS = 100
SIFT_FEATURES = 512


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
        if os.path.isdir(label_directory):
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

# Flatten the SIFT feature vectors
flattened_features = [feature for image_features in images for feature in image_features]

# Convert to numpy array
flattened_features = np.array(flattened_features)


# Perform feature scaling
scaler = StandardScaler()
flattened_features_scaled = scaler.fit_transform(flattened_features)

np.save("flattened_features_scaled.npy", flattened_features_scaled)
np.save("labels.npy",np.array(labels))
# Cluster the SIFT features to create a visual vocabulary
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, verbose=1)
kmeans.fit(flattened_features_scaled)


# Function to represent an image as a histogram of visual word occurrences
def image_to_histogram(image_features):
    # Assign each feature to its nearest cluster center
    visual_word_labels = kmeans.predict(image_features)

    # Create a histogram of visual word occurrences
    histogram, _ = np.histogram(visual_word_labels, bins=range(NUM_CLUSTERS + 1), density=True)

    return histogram


# Convert the images to histograms
histograms = []
for image_features in tqdm(images, desc='Converting images to histograms'):
    histogram = image_to_histogram(image_features)
    histograms.append(histogram)

# Convert the histograms and labels to numpy arrays
X = np.array(histograms)
y = np.array(labels)

# Save X and y as NumPy binary files
np.save("X.npy", X)
np.save("y.npy", y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=42)

# Train the MLP classifier
mlp.fit(X_train, y_train)

# Make predictions on the training and testing data
train_predictions = mlp.predict(X_train)
test_predictions = mlp.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

# Print the accuracy scores
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Plot the training loss and accuracy
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(mlp.loss_curve_)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(mlp.validation_scores_)
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

# Save the trained model
model_filename = "bovw_model.sav"
joblib.dump(mlp, model_filename)
