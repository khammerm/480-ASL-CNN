import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.models import load_model

## Create our flask app and load our trained model
app = Flask(__name__)
model = tf.keras.models.load_model('../ASL_480_CNN.h5')

# GET method to display our webpage
@app.route('/', methods=["GET"])
def hello_world():
    return render_template('index.html')

# POST method to grab our images locally and make a prediction
@app.route('/', methods=['POST'])
def predict():
    
    # Label dictionary, same as in our kerasCNN.py
    label_map = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
    'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14,
    'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21,
    'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26, 'nothing': 27, 'space': 28
    }
    
    # Make a list of all of our keys to predict a letter
    key_list = list(label_map.keys())
    
    # Getting images from webpage, then saving locally
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    
    # Loading image from local storage, then resizing for our CNN
    image = load_img(image_path, target_size=(64, 64))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)

    # array to find our highest % prediction, then append our prediction to the array
    label=[]
    label.append(key_list[np.argmax(yhat)])
    
    # for i in range(len(key_list)):
    #     if(label[i]) == label[0]:
    #         label.append(key_list[i])
    #         break
    
    # classification = our final prediction
    classification = label[0]

    # pass classification var to frontend
    return render_template('index.html', prediction=classification)
    
# running locally on port 3000
if __name__ == '__main__':
    app.run(port=3000, debug=True)