from flask import request, jsonify, Flask, render_template
from waitress import serve

import base64
import numpy as np
import io
import os

import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.utils.generic_utils import CustomObjectScope

import tensorflow as tf

from PIL import Image as view_image

app = Flask(__name__)

def get_model():
    global model, graph
    model_path = 'models/120_dog_breeds_classifier_v2.h5'

    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        model = load_model(model_path)

    graph = tf.get_default_graph()
    print('[INFO] Model loaded!')

def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array_expanded_dims = np.expand_dims(img_array, axis = 0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

print('[INFO] Loading Keras model...')
get_model()

predict_count = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    print('[INFO] Loading method...')
    num_of_result = 6
    message = request.get_json(force = True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = view_image.open(io.BytesIO(decoded))
    preprocessed_image = prepare_image(image, target_size=(224, 224))

    with graph.as_default():
        prediction = model.predict(preprocessed_image)
    
    result_dictionary = {}
    class_list = []
    class_index_count = 0
    print_count = 0

    class_array = []
    percentage_array = []

    global predict_count

    for folder in os.listdir('classes/120-dog-breeds/'):
        class_list.append(folder)

    for i in class_list:
        result_dictionary[i] = prediction[0][class_index_count]
        class_index_count = class_index_count + 1

    if num_of_result == 0:
        num_of_result = len(result_dictionary)

    print('[INFO] Loading results...')
    for key, value in sorted(result_dictionary.items(), key=lambda item: item[1], reverse = True):

        if print_count < num_of_result:
            class_array.append(key[10:])
            percentage_array.append(str(np.around(value*100, decimals = 2)) + "%")
            print_count = print_count + 1

    response = {
        'prediction': {
            'class': class_array,
            'percentage': percentage_array
        }
    }

    predict_count = predict_count + 1
    print('Number of predicts so far:', predict_count)

    return jsonify(response)

serve(app, host='0.0.0.0')