# from io import BytesIO
from fastai.vision.all import *
# import requests
# from PIL import Image
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def predict_image(image_url):
    # Download the image
    # response = requests.get(image_url)
    # with Image.open(BytesIO(response.content)) as img:
        
    img = PILImage.create(image_url)
    
    # Load the Fastai model
    learn = load_learner('./app/model/corn_model.pkl')

    # Make the prediction
    pred_class, pred_idx, outputs = learn.predict(img)

    # Return the prediction
    return pred_class
