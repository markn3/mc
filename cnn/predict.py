import cv2

class image_classifer:

    # constructor
    def __init__(self, model):
        # Instance variable
        self.model = model
        self.count = 0
    
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
        prediction = self.model.predict(prepped_img, verbose=0)
        # 30% threshold (subject to change)
        if(prediction[[0][0]] < 0.30):
            classification = "Cave"
        # calls cave count
        found  = self.cave_count(classification)
        return classification, prediction, found
    
    # if agent is in "cave" for 10 predictions then cave is found
    def cave_count(self, classification):
        if(classification == "Cave"):
            self.count += 1
        else:
            self.count = 0
        if(self.count == 10):
            return True
        else:
            return False
