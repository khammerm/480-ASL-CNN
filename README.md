## 480-ASL-CNN
Final project for COSC 480 - Neural Networks and Deep Learning. A Convolutional Neural Network that translates images of the American Sign Language to English

## How to run training / webdemo
I reccomend making 2 separate anaconda environments to prevent conflicting versions.
For training, download all required packages in **train-requirements.txt**
For the web demo, download all required packages in **webdemo-requirements.txt**

For training, run kerasCNN.py
For the web demo, run demo.py, then go to localhost:3000 to test.

## Dataset
@misc{https://www.kaggle.com/grassknoted/aslalphabet_akash nagaraj_2018,
title={ASL Alphabet},
url={https://www.kaggle.com/dsv/29550},
DOI={10.34740/KAGGLE/DSV/29550},

## References
https://github.com/grassknoted/Unvoiced/tree/master

## GitHub File Descriptions
**dataset**: test images for the web demo

**webdemo**: all related files for hosting basic flask app locally
  - **images**: where files uploaded to website are stored locally
  - **templates**: html for web page
  - **demo.py**: python file to host our webpage, run with python demo.py and check localhost.

**ASL_480_CNN.h5**: 95% accurate working model saved from tensorflow, used in web demo

**kerasCNN.py**: my final CNN file used for training.

**oldcnn.py**: old CNN I made to begin using Pytorch

**package-list**: list of all Anaconda and Pip packages, incase for some reason requirement files are giving issues

**train-requirements.txt**: list of all required packages for TRAINING, using kerasCNN.py

**webdemo-requirements.txt**: list of all required packages for the WEB DEMO, using demo.py

