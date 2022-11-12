print("Importing packages")
import tensorflow as tf
import cv2
import numpy as np


class image_classifer:

    # init method or constructor
    # constructor
    CATEGORIES = ["cave", "not_cave"]


    def __init__(self, model):
        # Instance variable
        self.model = model

    def prepare(img):
        return img.reshape(1, 360, 640, 3)

    def prepare_path(self, filepath):
        print(filepath)
        image = cv2.imread(filepath)
        new_array = cv2.resize(image, (360, 640))
        return new_array.reshape(1, 360, 640, 3)

    def classify(self, prepped_img):
        prediction = self.model.predict(prepped_img)
        classification = np.argmax(prediction,axis=1)
        return classification, prediction
