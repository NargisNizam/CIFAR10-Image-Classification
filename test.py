import tensorflow as tf
import numpy as np

# CIFAR-10 dataset load
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize images
x_test = x_test / 255.0

# Load trained model
model = tf.keras.models.load_model("model.keras")

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)