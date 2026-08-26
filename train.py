import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# CIFAR-10 dataset load
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Training images shape:", x_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing images shape:", x_test.shape)
print("Testing labels shape:", y_test.shape)


# CIFAR-10 class names
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


# Display 10 sample images
plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis("off")

plt.tight_layout()
plt.show()


# Count images in each class
class_counts = np.bincount(y_train.flatten())

print("\nClass Distribution:")

for i in range(10):
    print(f"{class_names[i]}: {class_counts[i]}")


# Class distribution graph
plt.figure(figsize=(10, 5))

plt.bar(class_names, class_counts)

plt.title("CIFAR-10 Class Distribution")
plt.xlabel("Classes")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
# Data Preprocessing
# Pixel values 0-255 ko 0-1 range mein convert karna
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("\nAfter Normalization:")
print("Minimum pixel value:", x_train.min())
print("Maximum pixel value:", x_train.max())
# CNN Model
model = tf.keras.Sequential([
    
    tf.keras.layers.Conv2D(
        32, (3, 3),
        activation="relu",
        input_shape=(32, 32, 3)
    ),
    
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64, (3, 3),
        activation="relu"
    ),
    
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        128, (3, 3),
        activation="relu"
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(10, activation="softmax")
])
# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
# Display model architecture
model.summary()

# Train the CNN model
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)
# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test Accuracy (%):", test_accuracy * 100)

# Accuracy Graph
plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("CNN Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("accuracy_graph.png", dpi=300)
plt.show()
# Loss Graph
plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("CNN Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("loss_graph.png", dpi=300)
plt.show()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate predictions
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Convert true labels to 1D
y_true = y_test.flatten()

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Blues", xticks_rotation=45)

plt.title("CIFAR-10 Confusion Matrix")
plt.tight_layout()
plt.show()
# Display sample predictions
plt.figure(figsize=(12, 8))

for i in range(12):
    plt.subplot(3, 4, i + 1)

    plt.imshow(x_test[i])

    predicted_label = class_names[y_pred_classes[i]]
    actual_label = class_names[y_true[i]]

    plt.title(
        f"Actual: {actual_label}\nPredicted: {predicted_label}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()
# Save the trained model
model.save("model.keras")

print("\nModel saved successfully as model.keras")