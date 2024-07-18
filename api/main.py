import numpy as np
from fastapi import FastAPI, UploadFile, File
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras import models, layers

app = FastAPI()
def create_model():
    BATCH_SIZE = 16
    IMAGE_SIZE = 256
    resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(256, 256),
    layers.experimental.preprocessing.Rescaling(1. / 255),
    ])
    input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)

    model = models.Sequential([
        resize_and_rescale,
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    model.build(input_shape=input_shape)
    return model

try:
    model = create_model()
    model.load_weights('../potatoes.h5')
    MODEL = model

except Exception as e:
    print(f"Error loading model: {e}")


CLASS_NAMES = ["Early Blight", "Light Blight", "Healthy"]
@app.get("/ping")
async def ping():
    return True

def readFileAsImage(data) -> np.ndarray:
    return np.array(Image.open(BytesIO(data)))


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image =readFileAsImage(await file.read())
    img_batch = np.expand_dims(image, axis=0)
    prediction = MODEL.predict(img_batch)
    class_predict = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))
    return {
        "class": class_predict,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)