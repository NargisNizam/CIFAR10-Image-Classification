import tensorflow as tf
import numpy as np

# CIFAR-10 dataset
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

# Load model
model = tf.keras.models.load_model("model.keras")

# Select one image
index = 0
image = x_test[index]

# Prediction
prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)

predicted_class = np.argmax(prediction)
actual_class = y_test[index][0]

print("Predicted:", class_names[predicted_class])
print("Actual:", class_names[actual_class])