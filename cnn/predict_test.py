print("Importing packages")
import tensorflow as tf
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
from predict import image_classifer

# load model into image_classifier instance
ic = image_classifer(tf.keras.models.load_model("cave_classifier.model"))

# path to test images
test_path = "C:\\Users\\markn\\Desktop\\mc\\CNN\\cnn\\test"
onlyfiles = [f for f in listdir(test_path) if isfile(join(test_path, f))]

# loop through images
for img in onlyfiles:
    print(img)
    img_path = "C:\\Users\\markn\\Desktop\\mc\\CNN\\cnn\\test\\" + str(img)
    print("Path: ", img_path)
    prepared_img = ic.prepare_path(img_path)
    classification, prediction = ic.classify(prepared_img)
    print("Prediction: ", prediction)
    print("Classification: ", classification, "\n\n")
