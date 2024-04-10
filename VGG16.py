import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Replace these paths with your actual dataset paths
DATASET_PATH = '/path/to/CAFE_dataset'

# Constants
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
NUM_CLASSES = 6
EPOCHS = 50

# Load and preprocess data using data loader technique
def load_data_with_loader(dataset_path, subset='training'):
    dataloader = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
        preprocessing_function=preprocess_input  # Fix: Import preprocess_input from VGG16
    )
    data_generator = dataloader.flow_from_directory(
        dataset_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset=subset
    )
    return data_generator


# Load dataset and split into train and validation sets
train_data_loader = load_data_with_loader(DATASET_PATH, subset='training')
test_data_loader = load_data_with_loader(DATASET_PATH, subset='validation')

# Split the training data loader into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    train_data_loader[0][0],  # Features (images)
    train_data_loader[0][1],  # Labels (one-hot encoded)
    test_size=0.2,
    random_state=42
)

# Load pre-trained VGG-16 model (without the top classification layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of the pre-trained model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the plotting function for loss and accuracy
def plot_metrics(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Train the model on the training dataset
history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val)
)


# Plot loss and accuracy
plot_metrics(history)


# Calculate testing accuracy
def calculate_test_accuracy(data_loader):
    y_true = data_loader.classes
    y_pred_probs = model.predict(data_loader)
    y_pred = np.argmax(y_pred_probs, axis=1)
    test_accuracy = accuracy_score(y_true, y_pred)
    return test_accuracy


# Calculate testing accuracy
test_accuracy = calculate_test_accuracy(test_data_loader)
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Calculate and plot confusion matrix
y_true = test_data_loader.classes
y_pred_probs = model.predict(test_data_loader)
y_pred = np.argmax(y_pred_probs, axis=1)
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# You can visualize the confusion matrix using the following code:
plt.imshow(conf_matrix, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

