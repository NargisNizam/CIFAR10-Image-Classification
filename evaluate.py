import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Load CIFAR-10 test dataset
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize images
x_test = x_test / 255.0

# Class names
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# Load trained model
model = tf.keras.models.load_model("model.keras")

# Make predictions
predictions = model.predict(x_test, verbose=1)

# Convert probabilities to class numbers
y_pred = np.argmax(predictions, axis=1)
y_true = y_test.flatten()

# Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))