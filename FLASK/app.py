from __future__ import division, print_function
import sys
import os
import glob
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model , model_from_json
from keras import backend
from tensorflow.keras import backend
import tensorflow as tf
global graph
graph=tf.compat.v1.get_default_graph()
from skimage.transform import resize
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from PIL import Image
import base64
import io

# Define a flask app
app = Flask(__name__)
# Load your trained model
model = load_model("C:/Users/user/Desktop/Pneumonia-Detection-Using-X-Rays-Using-Watson-Studio/PROJECT/FLASK/pneumonia.h5")
# Necessary
# print('Model loaded. Start serving...')
# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50 #model = ResNet50(weights='imagenet') #model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('base.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        im = Image.open(file_path)
        data = io.BytesIO()
        im.save(data, "JPEG")
        encoded_img_data = base64.b64encode(data.getvalue())
        
        with graph.as_default():
            json_file = open('model.json','r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            #load weights into new model
            loaded_model.load_weights("pneumonia.h5")
            loaded_model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['acc'])
            
            preds = loaded_model.predict(x)
            preds = preds > 0.5
        if preds[0][0]==0:
            text = "You are perfectly fine"
        else:
            text = "You are infected! Please Consult Doctor"
        return render_template("predict.html", value = text, image = encoded_img_data.decode('utf-8'))

if __name__ == '__main__':
    app.run(debug=True,threaded = False)
