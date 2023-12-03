import tensorflow as tf
import numpy as np

from flask import Flask, render_template, request
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.models import load_model

app = Flask(__name__)
model = tf.keras.models.load_model('../ASL_480_CNN.h5')

@app.route('/', methods=["GET"])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    
    
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    
    image = load_img(image_path, target_size=(64, 64))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    # prediction = model.prediction(image)
    
    prediction = np.argmax(yhat[0])
    
    label = decode_predictions(prediction)
    label = label[0][0]

    classification = '%s (%.2f%%)' % (label[1], label[2]*100)


    return render_template('index.html', prediction=classification)
    
    
    
if __name__ == '__main__':
    app.run(port=3000, debug=True)