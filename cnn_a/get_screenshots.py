# Importing all necessary libraries
import cv2
from os import listdir
from os.path import isfile, join

# Download videos using the basalt Behavioral Cloning folder
#       utils/download_dataset.py

# Path to folder containing videos
CAVE_PATH = "C:\\Users\\markn\\Desktop\\mc\\CNN\\MineRLBasaltFindCave-v0"
onlyfiles = [f for f in listdir(CAVE_PATH) if isfile(join(CAVE_PATH, f))]

current_vid = 0
# loop through videos
for vid in onlyfiles:
    if current_vid == 10:
        break
    print("current video: ", vid)
    
    # Path to current video
    VID_PATH = "C:\\Users\\markn\\Desktop\\mc\\CNN\\MineRLBasaltFindCave-v0\\" + str(vid)
    # Read the video from specified path
    cam = cv2.VideoCapture(VID_PATH)
    # Total length (in frames) of video
    length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Length of video: ", length)

    # Set video to certain time (starting from frame 100)
    cam.set(cv2.CAP_PROP_POS_FRAMES, 300)

    # Current frame of the video
    currentframe = 0
    while(True):
        # reading from frame
        ret,frame = cam.read()
        if ret:

            # Stop video after 100 frames
            if(currentframe == 500):
                break

            # Save every 10 frames
            if(currentframe % 50 == 0):
                # if video is still left continue creating images

                name = './data/frame' + str(currentframe) + '_' +str(current_vid) + '.jpg'
                print ('Creating...' + name)

                # writing the extracted images
                cv2.imwrite(name, frame)

                # increasing counter so that it will
                # show how many frames are created

            # next frame
            currentframe += 1
        else:
            break
    
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

    # next vid
    current_vid += 1