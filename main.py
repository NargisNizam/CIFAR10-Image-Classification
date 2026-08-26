from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import tensorflow as tf
import numpy as np
import io

app = FastAPI(title="CIFAR-10 Image Classifier")

templates = Jinja2Templates(directory="templates")

# Load trained model
model = tf.keras.models.load_model("model.keras")

# CIFAR-10 classes
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


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...)
):
    # Read uploaded image
    image_data = await file.read()

    # Open image
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Resize to CIFAR-10 image size
    image = image.resize((32, 32))

    # Convert to NumPy array
    image_array = np.array(image).astype("float32") / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array, verbose=0)

    predicted_class = np.argmax(prediction[0])
    confidence = float(np.max(prediction[0])) * 100

    result = class_names[predicted_class]

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": result,
            "confidence": round(confidence, 2)
        }
    )