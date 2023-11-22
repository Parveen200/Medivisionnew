import os
import tensorflow as tf
import numpy as np

from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


# Define a flask app
Pneumonia_app = Flask(__name__)


model = load_model('PneumoniaDetectionModel.h5')
print('Model loaded. Check http://127.0.0.1.5000/')


def pneumonia_get_className(classNo):
    if classNo==0:
        return "No it's Normal"
    elif classNo==1:
        return "Yes it's Pneumonia"


def pneumonia_getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64,64))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=model.predict(input_img) # type: ignore
    return result


@Pneumonia_app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('Pneumonia.html')


@Pneumonia_app.route('/predict', methods=['GET', 'POST']) # type: ignore
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']


        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename)) # type: ignore
        f.save(file_path)
        value=pneumonia_getResult(file_path)
        result=pneumonia_get_className(value)
        return result
    return None


if __name__=='__main__':
    Pneumonia_app.run(debug=True)