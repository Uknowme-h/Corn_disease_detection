from fastapi import FastAPI
from pydantic import BaseModel
from model.model import predict_image


app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello, World"}