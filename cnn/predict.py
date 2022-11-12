print("Importing packages")
import tensorflow as tf
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np


CATEGORIES = ["cave", "not_cave"]


def prepare(filepath):
    print(filepath)
    image = cv2.imread(filepath)
    new_array = cv2.resize(image, (360, 640))
    return new_array.reshape(1, 360, 640, 3)

# image = np.array(Image.open("path").resize(256,256,3))
# load model
model = tf.keras.models.load_model("cave_classifier_1111.model")

test_path = "C:\\Users\\markn\\Desktop\\mc\\CNN\\data\\test"
onlyfiles = [f for f in listdir(test_path) if isfile(join(test_path, f))]

current_img = 0
# loop through images
for img in onlyfiles:
    print(img)
    img_path = "C:\\Users\\markn\\Desktop\\mc\\CNN\\data\\test\\" + str(img)
    prediction = model.predict(prepare(img_path))
    classes_x=np.argmax(prediction,axis=1)
    print("Prediction: ", prediction)
    print("Result: ", classes_x)

    if(int(prediction[0][0] > int(prediction[0][1])) and prediction[0][0] >= 20):
        print(CATEGORIES[0])
    else:
        print(CATEGORIES[1])




# img = env.render(mode="rgb_array")
# returns the rendered image
