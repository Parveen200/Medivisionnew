from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename  
from BrainTumor_app import get_className, getResult
from BreastCancer_app import breast_get_className, breast_getResult
from Pneumonia_app import pneumonia_get_className, pneumonia_getResult


app = Flask(__name__, static_folder='static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Braintumor')
def braintumor():
    return render_template('Braintumor.html')

@app.route('/Breastcancer')
def breastcancer():
    return render_template('Breastcancer.html')

@app.route('/Pneumonia')
def pneumonia():
    return render_template('Pneumonia.html')


@app.route('/predict', methods=['POST'])  # type: ignore
def predict_braintumor():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))  # type: ignore
        f.save(file_path)
        value = getResult(file_path)
        result = get_className(value)
        return result

    return render_template("/predict")




@app.route('/predict_breastcancer', methods=['POST'])  # type: ignore
def predict_breastcancer():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))  # type: ignore
        f.save(file_path)
        value = breast_getResult(file_path)
        result = breast_get_className(value)
        return result

    return render_template("/predict")



@app.route('/predict_pneumonia', methods=['POST']) # type: ignore
def predict_pneumonia():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename)) # type: ignore
        f.save(file_path)
        value = pneumonia_getResult(file_path)
        result = pneumonia_get_className(value)
        return result

    return render_template("/predict")

if __name__ == '__main__':
    app.run(debug=True)
