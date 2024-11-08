

import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Set paths to the data directories
train_dir = r"E:\UIDATAA\train"
test_dir = r"E:\UIDATAA\test"

# Parameters
n_clusters = 50  # Number of clusters for BoVW
image_size = (128, 128)  # Resize images for CNN

def extract_orb_features(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if descriptors is None:
        descriptors = np.array([])
    return descriptors

def load_images_and_extract_features(directory):
    descriptors_list = []
    labels = []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if not os.path.isdir(label_dir):
            continue
        for filename in os.listdir(label_dir):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(label_dir, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                # Preprocessing
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_image = clahe.apply(image)
                denoised_image = cv2.bilateralFilter(enhanced_image, 9, 75, 75)
                thresholded_image = cv2.adaptiveThreshold(denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                          cv2.THRESH_BINARY, 11, 2)
                sobelx = cv2.Sobel(thresholded_image, cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(thresholded_image, cv2.CV_64F, 0, 1, ksize=5)
                edges = cv2.magnitude(sobelx, sobely)

                # Feature extraction
                descriptors = extract_orb_features(np.uint8(edges))
                if descriptors.size != 0:
                    descriptors_list.append(descriptors)
                    labels.append(label)
    return descriptors_list, labels

def build_bovw(descriptors_list, n_clusters):
    descriptors_stack = np.vstack(descriptors_list)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(descriptors_stack)
    return kmeans

def extract_features_bovw(descriptors_list, kmeans, n_clusters):
    features = []
    for descriptors in descriptors_list:
        histogram = np.zeros(n_clusters)
        if descriptors is not None and descriptors.size != 0:
            cluster_result = kmeans.predict(descriptors)
            for i in cluster_result:
                histogram[i] += 1
        features.append(histogram)
    return np.array(features)

def augment_images(directory, image_size=(128, 128)):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if not os.path.isdir(label_dir):
            continue
        for filename in os.listdir(label_dir):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(label_dir, filename)
                image = cv2.imread(image_path)
                if image is None:
                    continue
                image = cv2.resize(image, image_size)
                image = np.expand_dims(image, axis=0)
                for batch in datagen.flow(image, batch_size=1):
                    break
                augmented_image = batch[0].astype(np.uint8)
                augmented_path = os.path.join(label_dir, 'aug_' + filename)
                cv2.imwrite(augmented_path, augmented_image)

# Load and process training data
print("Processing training data...")
train_descriptors_list, train_labels = load_images_and_extract_features(train_dir)

# Augment training images
print("Augmenting training images...")
augment_images(train_dir, image_size=image_size)

# Load and process augmented training data
print("Processing augmented training data...")
aug_train_descriptors_list, aug_train_labels = load_images_and_extract_features(train_dir)
train_descriptors_list.extend(aug_train_descriptors_list)
train_labels.extend(aug_train_labels)

# Build BoVW model
print("Building BoVW model...")
kmeans = build_bovw(train_descriptors_list, n_clusters)

# Extract BoVW features for training data
print("Extracting BoVW features for training data...")
X_train_bovw = extract_features_bovw(train_descriptors_list, kmeans, n_clusters)

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_labels)

# Scale features
scaler = StandardScaler()
X_train_bovw = scaler.fit_transform(X_train_bovw)

# Set up Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=7,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='log2',
    bootstrap=True,
    random_state=42
)

# Evaluate Random Forest with cross-validation
print("Evaluating Random Forest with cross-validation...")
cv_scores = cross_val_score(rf, X_train_bovw, y_train, cv=skf)
print(f'Cross-Validation Accuracy Scores: {cv_scores}')
print(f'Average Cross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}%')

# Train the Random Forest model
rf.fit(X_train_bovw, y_train)

# Process test data
print("Processing testing data...")
test_descriptors_list, test_labels = load_images_and_extract_features(test_dir)
X_test_bovw = extract_features_bovw(test_descriptors_list, kmeans, n_clusters)
X_test_bovw = scaler.transform(X_test_bovw)
y_test = le.transform(test_labels)

# Predict and evaluate Random Forest on test set
y_pred_rf = rf.predict(X_test_bovw)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Test Set Accuracy: {accuracy_rf * 100:.2f}%')
print("Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# Define the CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Data augmentation for CNN
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.1
)

# Load and preprocess training images for CNN
def load_and_preprocess_images(directory, image_size):
    images = []
    labels = []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if not os.path.isdir(label_dir):
            continue
        for filename in os.listdir(label_dir):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(label_dir, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                image = cv2.resize(image, image_size)
                image = np.expand_dims(image, axis=-1)  # Add this line to expand dimensions
                images.append(image)
                labels.append(label)
    return np.array(images), labels

print("Processing training data for CNN...")
X_train_cnn, y_train_cnn_labels = load_and_preprocess_images(train_dir, image_size)
y_train_cnn = le.transform(y_train_cnn_labels)
y_train_cnn = to_categorical(y_train_cnn, num_classes=len(np.unique(y_train_cnn)))

print("Processing testing data for CNN...")
X_test_cnn, y_test_cnn_labels = load_and_preprocess_images(test_dir, image_size)
y_test_cnn = le.transform(y_test_cnn_labels)
y_test_cnn = to_categorical(y_test_cnn, num_classes=len(np.unique(y_test_cnn)))

# Normalize images for CNN
X_train_cnn = X_train_cnn / 255.0
X_test_cnn = X_test_cnn / 255.0

# Define CNN model
input_shape = (image_size[0], image_size[1], 1)  # Grayscale images
cnn_model = create_cnn_model(input_shape, num_classes=len(le.classes_))

# Compile and train CNN model
cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train with data augmentation
batch_size = 32
cnn_history = cnn_model.fit(
    datagen.flow(X_train_cnn, y_train_cnn, batch_size=batch_size, subset='training'),
    validation_data=datagen.flow(X_train_cnn, y_train_cnn, batch_size=batch_size, subset='validation'),
    epochs=50
)

# Evaluate CNN model
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test_cnn)
print(f'CNN Test Accuracy: {cnn_accuracy * 100:.2f}%')

# Plot CNN training history
plt.plot(cnn_history.history['accuracy'], label='Train Accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('CNN Training Accuracy vs Validation Accuracy')
plt.legend()
plt.show()

#performace matrix
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# Generate predictions on the test set
y_pred_cnn = cnn_model.predict(X_test_cnn)
y_pred_classes = np.argmax(y_pred_cnn, axis=1)  # Convert one-hot encoded to class indices
y_test_classes = np.argmax(y_test_cnn, axis=1)

# Classification report
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=le.classes_))

# Confusion matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import roc_curve, auc, RocCurveDisplay

# Get the predicted probabilities for each class
y_pred_proba = cnn_model.predict(X_test_cnn)

# Plot ROC curve for each class
fpr = {}
tpr = {}
roc_auc = {}
for i in range(len(le.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test_cnn[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotting the ROC curves
plt.figure(figsize=(10, 8))
for i in range(len(le.classes_)):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve for {le.classes_[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guessing
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()




le = LabelEncoder()
y_train_cnn = le.fit_transform(y_train_cnn_labels)  # Numeric labels
y_test_cnn = le.transform(y_test_cnn_labels)  # Transform test labels

# To see the mapping of class names to numeric labels:
class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Class Mapping: ", class_mapping) 

import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

# Function to preprocess a single image for prediction (convert to grayscale)
def preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
    image = cv2.resize(image, target_size)  # Resize to match the input size of the model
    image = image.astype('float') / 255.0  # Normalize the image
    image = img_to_array(image)  # Convert image to an array
    image = np.expand_dims(image, axis=-1)  # Add the channel dimension for grayscale (128, 128, 1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension (for a single image)
    return image

# Function to predict the class of a single image
def predict_single_image(cnn_model, image_path, label_encoder):
    processed_image = preprocess_image(image_path)
    prediction = cnn_model.predict(processed_image)  # Get prediction
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the numeric class (0 or 1)
    class_name = label_encoder.inverse_transform([predicted_class])[0]  # Convert numeric class to class name
    return class_name, prediction

# Example usage:
image_path = r"E:\test set\infected\img_0_53.jpg"
predicted_class_name, prediction = predict_single_image(cnn_model, image_path, le)

# Print results
print(f'Predicted class name: {predicted_class_name}')  # Prints 'infected' or 'notinfected'
print(f'Prediction probabilities: {prediction}')  # Prints the probabilities for each class
