import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load CIFAR-10 test dataset
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize
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

# Predictions
predictions = model.predict(x_test, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = y_test.flatten()

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(xticks_rotation=45)
plt.title("CIFAR-10 Confusion Matrix")
plt.tight_layout()

# Save image
plt.savefig("confusion_matrix.png", dpi=300)

print("Confusion Matrix saved successfully!")