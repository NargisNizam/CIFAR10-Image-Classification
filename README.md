# CIFAR-10 Image Classification using CNN

## 📌 Project Overview

This project implements an image classification model using a Convolutional Neural Network (CNN) and the CIFAR-10 dataset.

The model classifies images into 10 different categories such as airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## 📊 Dataset

CIFAR-10 contains:

- 50,000 training images
- 10,000 testing images
- Image size: 32 × 32 pixels
- Number of classes: 10

### Classes

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## 🧠 Model Architecture

The CNN model contains:

- Conv2D Layer – 32 filters
- MaxPooling2D
- Conv2D Layer – 64 filters
- MaxPooling2D
- Conv2D Layer – 128 filters
- Flatten Layer
- Dense Layer – 128 neurons
- Dropout – 0.5
- Output Layer – 10 classes

## ⚙️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn

## 🔄 Data Preprocessing

Pixel values were normalized from the range:

`0–255`

to:

`0–1`

## 📈 Model Training

The model was trained for 10 epochs with:

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: 64
- Validation Split: 20%

## 📊 Evaluation

The model was evaluated using:

- Test Accuracy
- Test Loss
- Classification Report
- Confusion Matrix
- Sample Predictions

## 🖼️ Sample Prediction

The trained model successfully predicted sample CIFAR-10 images.

Example:

**Predicted:** Cat  
**Actual:** Cat

## 📁 Project Structure

```text
CIFAR10-Image-Classification/
│
├── train.py
├── test.py
├── predict.py
├── evaluate.py
├── confusion_matrix.py
├── model.keras
├── accuracy_graph.png
├── loss_graph.png
├── confusion_matrix.png
└── README.md
🚀 How to Run
1. Install dependencies
pip install tensorflow numpy matplotlib scikit-learn
2. Train the model
python train.py
Test the model
python test.py
4. Make predictions
python predict.py
5. Generate evaluation results
python evaluate.py
👩‍💻 Author

Nargis Nizam

AI & Data Science Student

