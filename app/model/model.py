from fastai import *
from fastai.vision import *
from PIL import Image

def predict_image(image_path):
    # Load the Fastai model
    learn = load_learner('/path/to/model.pkl')

    # Open the image using PIL
    image = Image.open(image_path)

    # Preprocess the image
    img_tensor = pil2tensor(image, dtype=np.float32)
    img_tensor.div_(255)

    # Make the prediction
    pred_class, pred_idx, outputs = learn.predict(Image(img_tensor))

    # Return the prediction
    if pred_class == 'healthy':
        return 'healthy'
    else:
        return 'infected'
