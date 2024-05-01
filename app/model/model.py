from fastai import *
from fastai.vision.all import *


def predict_image(image_path):
    # Load the Fastai model
    learn = load_learner('./model/corn_model.pkl')

    img = PILImage.create(image_path)

    # Make the prediction
    pred_class, pred_idx, outputs = learn.predict(img)

    # Return the prediction
    return pred_class

#  Set-ExecutionPolicy Unrestricted -Scope Process