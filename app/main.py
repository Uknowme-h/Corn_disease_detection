from fastapi import FastAPI
from app.model.model import predict_image

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello, World"}

@app.get("/predict/")
def predict():
    # if not image_url:
    #     return {"error": "image_url is required"}
    
    try:
        prediction = predict_image('./app/model/example2.jpg')
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}
