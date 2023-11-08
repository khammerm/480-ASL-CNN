import os                                                               # os library for loading and iterating through our images
import random                                                           # may use this at a later time        
import pandas as pd
import numpy as np
import skimage                                                          # Scikit-image library
import cv2                                                              # Computer vision library for reading images
import tensorflow as tf                                                 # Using tensorflow + keras for our model
from tensorflow import keras                                                      
from skimage.transform import resize                            
from sklearn.model_selection import train_test_split                    # This function helps split our data into training / testing
from tensorflow.keras.optimizers import Adam                            # Adam optimizer
from tensorflow.keras.utils import to_categorical                       # One-Hot encoding w/ this API
from tensorflow.keras.callbacks import EarlyStopping                    # Early Stopping to prevent overfitting method *WIP*
from keras.models import Sequential                                     # keras.models + layers for building our CNN
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten, BatchNormalization, Dropout
from sklearn.metrics import classification_report, confusion_matrix     # Saw other CNN's using these for analysis *WIP*



# Feedback that we had no issues importing any of our packages
print("packages imported")
print(tf.test.gpu_device_name())
TRAIN_DATA_PATH = "./dataset/asl_alphabet_train"
TEST_DATA_PATH = "./dataset/asl_alphabet_test"

# Pre defined variables for network
batch_size = 64
imageSize = 64
target_dims = (imageSize, imageSize, 3)
num_classes = 29
train_len = 87000

# Create a dictionary to map folder names to labels
# We actually have 29 labels here, 26 for alphabet, and 3 special characters (del, nothing, space)
label_map = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
    'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14,
    'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21,
    'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26, 'nothing': 27, 'space': 28
}

# Helper function for loading images
def load_image(file_path):
    img = cv2.imread(file_path)
    if img is None:
        return None
    # Resizing images to 64x64, 3 input channels since color images
    img = skimage.transform.resize(img, (imageSize, imageSize, 3))
    return img

def get_data(folder, label_map):
    # initialize empty arrays to hold our data (x) and our labels (y)
    x = []
    y = []
    
    # iterate through all folders in our directory, specified in TRAIN_DATA_PATH
    for folderName in os.listdir(folder):
        # This is for testing purposes
        print(folderName)
        # Any folder not in our label map is not needed, if statement to catch
        if folderName not in label_map:
            continue 
        label = label_map[folderName]
        folder_path = os.path.join(folder, folderName)
        
        # Reading through each file
        for image_filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_filename)
            img = load_image(image_path)
            # if our img exists, append to our data arr.
            if img is not None:
                x.append(img)
                y.append(label)
    # return our array        
    return np.array(x), np.array(y)
# Need to figure out how to use my GPU for training / loading images. Runtime is a major concern here.

x_train, y_train = get_data(TRAIN_DATA_PATH, label_map)
    
print("Images successfully imported")

#Creating new variables for our train/test split
x_data = x_train
y_data = y_train

# train_test_split from sklearn library
# test = .25 * size of our training library
# random_state = shuffling so we get random images for testing, not all from same group
# stratify = our labels. since our Y data is a dictionary of labels, we use this to 'stratify' our data
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size=0.35, random_state=42, stratify=y_data)

# One-Hot Encoding, 29 classes
y_onehot_train = to_categorical(y_train,29)
y_onehot_test = to_categorical(y_test,29)

# Printing our shapes to make sure everything was loaded correctly up to this point
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_onehot_train.shape)
print(y_onehot_test.shape)

# from Keras, using a sequential model to build our CNN
model = Sequential()

#Conv. Layer 1
model.add(Conv2D(32, (3,3), input_shape=(64,64,3))) # 32 feature maps, 3x3 size, input shape (64,64,3)
model.add(BatchNormalization())                     # Normalization function for our batch
model.add(Activation('relu'))                       # ReLu activation function for conv. layers
model.add(MaxPooling2D((2,2)))                      # 2x2 pooling size

#Conv. Layer 2
model.add(Conv2D(64, (3,3)))                        # 64 feature maps, 3x3 size, input shape (64,64,3)
model.add(BatchNormalization())                     # Normalization function for our batch
model.add(Activation('relu'))                       # ReLu activation function for conv. layers
model.add(MaxPooling2D((2,2)))                      # 2x2 pooling size

#Conv. Layer 3
model.add(Conv2D(128, (3,3)))                       # 128 feature maps, 3x3 size, input shape (64,64,3)
model.add(BatchNormalization())                     # Normalization function for our batch
model.add(Activation('relu'))                       # ReLu activation function for conv. layers
model.add(MaxPooling2D((2,2)))                      # 2x2 pooling size

model.add(Flatten())                                # Flatten our conv. layers in the model to connect to our FC layers

# Fully Connected Layer 1
model.add(Dense(256, activation='relu'))            # 128 nodes in our first FC layer, ReLu activation function
model.add(Dropout(0.5))                             # Dropout Layer that removes weights < .5 to help with overfitting

# Fully Connected Layer 2
model.add(Dense(29, activation='softmax'))          # 29 outputs, using softmax activation here

model.summary()                                     # this function shows a summary of our CNN up to this point

# Early stopping to prvent overfitting, since runtime is a concern *WIP*
# early_stop = EarlyStopping(monitor='val_loss',patience=2)

# Using Adam optimizer
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# params = training data, training labels, # of epochs, batch size, display progress (tensorflow), validation data for performance analysis
# Possibly add "Learning rate scheduler" to yield better results
# This would update our learning rate in real time as we get feedback from our optimizer.

model.fit(x_train, y_onehot_train, epochs = 3, batch_size=64, verbose=2, validation_data=(x_test,y_onehot_test))


# Would like to create a dataframe here to display the performance per epoch, need to figure out how to do it still
# metrics = pd.DataFrame(model.history.history)
# print("Model metrics:")

