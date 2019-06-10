from flask import request, jsonify, Flask, render_template

import base64
import numpy as np
import io

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
    model_path = 'static/assets/models/120_dog_breeds_classifier_v2.h5'

    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        model = load_model(model_path)

    graph = tf.get_default_graph()
    print('[INFO] Model loaded!')

def prepare_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis = 0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

print('[INFO] Loading Keras model...')
get_model()

predict_count = 0

class_list = [
    'n02085620-Chihuahua', 'n02085782-Japanese_spaniel', 'n02085936-Maltese_dog',
    'n02086079-Pekinese', 'n02086240-Shih-Tzu', 'n02086646-Blenheim_spaniel',
    'n02086910-papillon', 'n02087046-toy_terrier', 'n02087394-Rhodesian_ridgeback',
    'n02088094-Afghan_hound', 'n02088238-basset', 'n02088364-beagle',
    'n02088466-bloodhound', 'n02088632-bluetick', 'n02089078-black-and-tan_coonhound',
    'n02089867-Walker_hound', 'n02089973-English_foxhound', 'n02090379-redbone',
    'n02090622-borzoi', 'n02090721-Irish_wolfhound', 'n02091032-Italian_greyhound',
    'n02091134-whippet', 'n02091244-Ibizan_hound', 'n02091467-Norwegian_elkhound',
    'n02091635-otterhound', 'n02091831-Saluki', 'n02092002-Scottish_deerhound',
    'n02092339-Weimaraner', 'n02093256-Staffordshire_bullterrier', 'n02093428-American_Staffordshire_terrier',
    'n02093647-Bedlington_terrier', 'n02093754-Border_terrier', 'n02093859-Kerry_blue_terrier',
    'n02093991-Irish_terrier', 'n02094114-Norfolk_terrier', 'n02094258-Norwich_terrier',
    'n02094433-Yorkshire_terrier', 'n02095314-wire-haired_fox_terrier', 'n02095570-Lakeland_terrier',
    'n02095889-Sealyham_terrier', 'n02096051-Airedale', 'n02096177-cairn',
    'n02096294-Australian_terrier', 'n02096437-Dandie_Dinmont', 'n02096585-Boston_bull',
    'n02097047-miniature_schnauzer', 'n02097130-giant_schnauzer', 'n02097209-standard_schnauzer',
    'n02097298-Scotch_terrier', 'n02097474-Tibetan_terrier', 'n02097658-silky_terrier',
    'n02098105-soft-coated_wheaten_terrier', 'n02098286-West_Highland_white_terrier', 'n02098413-Lhasa',
    'n02099267-flat-coated_retriever', 'n02099429-curly-coated_retriever', 'n02099601-golden_retriever',
    'n02099712-Labrador_retriever', 'n02099849-Chesapeake_Bay_retriever', 'n02100236-German_short-haired_pointer',
    'n02100583-vizsla', 'n02100735-English_setter', 'n02100877-Irish_setter',
    'n02101006-Gordon_setter', 'n02101388-Brittany_spaniel', 'n02101556-clumber',
    'n02102040-English_springer', 'n02102177-Welsh_springer_spaniel', 'n02102318-cocker_spaniel',
    'n02102480-Sussex_spaniel', 'n02102973-Irish_water_spaniel', 'n02104029-kuvasz',
    'n02104365-schipperke', 'n02105056-groenendael', 'n02105162-malinois',
    'n02105251-briard', 'n02105412-kelpie', 'n02105505-komondor',
    'n02105641-Old_English_sheepdog', 'n02105855-Shetland_sheepdog', 'n02106030-collie',
    'n02106166-Border_collie', 'n02106382-Bouvier_des_Flandres', 'n02106550-Rottweiler',
    'n02106662-German_shepherd', 'n02107142-Doberman', 'n02107312-miniature_pinscher',
    'n02107574-Greater_Swiss_Mountain_dog', 'n02107683-Bernese_mountain_dog', 'n02107908-Appenzeller',
    'n02108000-EntleBucher', 'n02108089-boxer', 'n02108422-bull_mastiff',
    'n02108551-Tibetan_mastiff', 'n02108915-French_bulldog', 'n02109047-Great_Dane',
    'n02109525-Saint_Bernard', 'n02109961-Eskimo_dog', 'n02110063-malamute',
    'n02110185-Siberian_husky', 'n02110627-affenpinscher', 'n02110806-basenji',
    'n02110958-pug', 'n02111129-Leonberg', 'n02111277-Newfoundland',
    'n02111500-Great_Pyrenees', 'n02111889-Samoyed', 'n02112018-Pomeranian',
    'n02112137-chow', 'n02112350-keeshond', 'n02112706-Brabancon_griffon',
    'n02113023-Pembroke', 'n02113186-Cardigan', 'n02113624-toy_poodle',
    'n02113712-miniature_poodle', 'n02113799-standard_poodle', 'n02113978-Mexican_hairless',
    'n02115641-dingo', 'n02115913-dhole', 'n02116738-African_hunting_dog'
    ]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    print('[INFO] Loading identifier method...')
    num_of_result = 6
    message = request.get_json(force = True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = view_image.open(io.BytesIO(decoded))
    preprocessed_image = prepare_image(image, target_size=(224, 224))

    with graph.as_default():
        prediction = model.predict(preprocessed_image)
    
    result_dictionary = {}
    class_index_count = 0
    print_count = 0

    class_array = []
    percentage_array = []

    global predict_count
    global class_list

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

if __name__ == '__main__':
    app.run()