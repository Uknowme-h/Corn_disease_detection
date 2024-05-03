from fastapi import FastAPI, UploadFile
from app.model.model import predict_image
from fastapi import FastAPI, UploadFile, File
from fastai.vision.all import *
import shutil

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello, World"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Convert the uploaded file to PILImage
        img = PILImage.create(file.file)

        # Get predictions using the converted image
        prediction = predict_image(img)

        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}
