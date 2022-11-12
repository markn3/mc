print("Importing packages")
import tensorflow as tf
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
from predict import image_classifer

ic = image_classifer(tf.keras.models.load_model("cave_classifier_1111.model"))

CATEGORIES = ["cave", "not_cave"]

test_path = "C:\\Users\\markn\\Desktop\\mc\\CNN\\cnn\\test"
onlyfiles = [f for f in listdir(test_path) if isfile(join(test_path, f))]

current_img = 0
# loop through images
for img in onlyfiles:
    print(img)
    img_path = "C:\\Users\\markn\\Desktop\\mc\\CNN\\cnn\\test\\" + str(img)
    print("Path: ", img_path)
    prepared_img = ic.prepare_path(img_path)
    classification, prediction = ic.classify(prepared_img)
    print("Prediction: ", prediction)
    print("Result: ", classification)

    if(int(prediction[0][0] > int(prediction[0][1])) and prediction[0][0] >= 20):
        print(CATEGORIES[0])
    else:
        print(CATEGORIES[1])


