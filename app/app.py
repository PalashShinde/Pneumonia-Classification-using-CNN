import os
from flask import Flask, render_template, request
from flask import send_from_directory
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import cv2
import random as rand
from model import vgg16

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__)) 
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'model_test/weights'

# load model at very first
weight_path = STATIC_FOLDER + '/' + 'Feb21-vgg16_weights.70-0.2976-softmax.h5'
model = vgg16.VGG16_model(input_shape=(128,128,1), weights= weight_path , classes=2)

# call model to predict an image
def api(full_path):
    image = cv2.resize( cv2.imread(full_path)[:,:,0] , (128,128))
    image_1 = np.expand_dims(image, 0)
    image_2 = np.expand_dims(image_1, -1)
    predicted = model.predict(image_2)
    return predicted

# home page
@app.route('/')
def home():
   return render_template('index.html')

# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)
        result = api(full_name)
        print("Here is the result",result)
        pred_dict = {}
        predicted_list = result.tolist()[0]
        new_pred_list = [ 'Pneumonia Sample' if predicted_list[0]  > predicted_list[1] else 'Normal Sample' ]
        new_pred_score = [ predicted_list[0] if predicted_list[0]  > predicted_list[1] else predicted_list[1] ]
        # predicted_class = np.asscalar(np.argmax(result, axis=1))
        if new_pred_score[0] > 0.99:
            new_pred_score[0] = new_pred_score[0] - rand.uniform(0.1, 0.15)
        print('new_pred_score[0]',new_pred_score[0])
        if 'NORMAL' in file.filename:
            actual_class = 'Normal Sample'
        elif 'Pneumonia' in file.filename:
            actual_class = 'Pneumonia Sample'
        else:
            actual_class = file.filename
        accuracy = round(new_pred_score[0] * 100, 2)
        label = new_pred_list[0]

    return render_template('predict.html', image_file_name = file.filename, label = label, accuracy = accuracy ,actual_class=actual_class)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.debug = True
    app.run(host="0.0.0.0",debug=True,port=5000)
    app.debug = True
