# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

#avoid using GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#set image to process
testImg = "test samples/fruits1.jpg"

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", type=str,
                default="fruit_detection.model",
                help="path to trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")

ap.add_argument("-i", "--image", default=testImg, help = "")
args = vars(ap.parse_args())


# load the detector model from disk
print("[INFO] loading model...")
model = load_model(args["model"])

# load the input image from disk, clone it, and grab the image spatial
# dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Find contours, obtain bounding box, extract and save ROI
ROI_number = 0
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    #Make ROI image
    x,y,w,h = cv2.boundingRect(c)
    ROI = orig[y:y+h, x:x+w]
    ROI_number += 1

    #If size too small compared to original image
    if(w<image.shape[1]/10 and h<image.shape[1]/10):
        continue


    try:
        ROI = cv2.resize(ROI, (224, 224))
        ROI = img_to_array(ROI)
        ROI = preprocess_input(ROI)
        ROI = np.expand_dims(ROI, axis=0)
        (apple, banana) = model.predict(ROI)[0]

        if(max(apple, banana)<0.9):
            continue
        label = "Apple" if apple > banana else "Banana"
        color = (0, 255, 0)
        # color = (0, 255, 0) if label == "Apple" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(apple, banana) * 100)
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        print("Found")

    except:
        print("ROI Image ignored")


cv2.imshow('image', image)
cv2.waitKey()