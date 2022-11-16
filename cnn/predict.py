print("Importing packages")
import tensorflow as tf
import cv2
import numpy as np

class image_classifer:

    # constructor
    def __init__(self, model):
        # Instance variable
        self.model = model
    
    # resize image
    def prepare(img):
        return img.reshape(1, 360, 640, 3)

    # used only if images are in some directory
    def prepare_path(self, filepath):
        print(filepath)
        image = cv2.imread(filepath)
        new_array = cv2.resize(image, (360, 640))
        return new_array.reshape(1, 360, 640, 3)

    
    def classify(self, prepped_img):
        classification = "NotCave"
        prediction = self.model.predict(prepped_img)
        if(prediction[[0][0]] < 0.5):
            classification = "Cave"
        return classification, prediction
