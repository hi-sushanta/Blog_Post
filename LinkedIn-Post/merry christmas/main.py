import math
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
# Initialize cascaded classifier class.
face_cascade = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")


detector = HandDetector()


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

cake = cv2.resize(cv2.imread('merry-cristmass3.png',flags=cv2.IMREAD_COLOR),(612,408))
cake = cv2.flip(cake, 1)
santa_hat = cv2.resize(cv2.imread('head2.png',flags=cv2.IMREAD_COLOR),(612,408))
santa_beard = cv2.resize(cv2.imread('santa-claus-beard.png',flags= cv2.IMREAD_COLOR),(612,408))
snow_cap = cv2.VideoCapture("snow-fall2.mp4")
snow_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
snow_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

# Function for detecting faces in and image
def detectFaceOpenCVDnn(net, frame):
    # create a blot from image and some pre-processing
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    # Set the blob input to the model
    net.setInput(blob)
    detections = net.forward()
    return detections

# Function for annotating the image with bounding boxes for each detected face.
def process_detections(frame, detections, conf_threshold=0.5):
    bboxes = []
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    # Loop over all detection and draw bounding boxes around each faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            bboxes.append([x1, y1, x2, y2])
            c1 = int(x1 + (x2 - x1)/2)
            c2 = int(y1 + (y2 - y1)/2)
            x3 = c1
            y3 = c2 - (c2 - y1)
            x4 = c1 
            y4 = c2 + (y2 - c2)
           
            frame = Overlay(frame, santa_hat, x3-15,y3-15, 130)
            frame = Overlay(frame,santa_beard,x4,y4-40,90)

    return frame, bboxes
# Function to load the DNN model.
def load_model():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

def mapFromTo(x,a,b,c,d):
    return (x-a)/(b-a)*(d-c)+c

def Overlay (background, overlay, x, y, size):
    background_h, background_w, c = background.shape
    imgScale = mapFromTo(size, 200, 40, 1.3, 0.3)
    overlay = cv2.resize(overlay, (0, 0), fx=imgScale, fy=imgScale)
    h, w, c = overlay.shape
    try:
        if x + w/2 >= background_w or y + h/2 >= background_h or x - w/2 <= 0 or y - h/2 <= 0:
            return background
        else:
            overlayImage = overlay[..., :3]
            mask = (overlay / 255.0)
            background[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = (1-mask)*background[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] + overlayImage
            return background
    except:
        return background


def findDistance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

net = load_model()

while True:
    success, img = cap.read()
    success2,img2 = snow_cap.read()
    if success2:
        # print('hello')
        img2 = cv2.resize(img2,(img.shape[1],img.shape[0]))
        final = cv2.addWeighted(img2, 0.3, img, 0.7, 0) 
    elif success2 == False:
        snow_cap = cv2.VideoCapture('snow-fall.mp4')
        success2,img2 = snow_cap.read()
        img2 = cv2.resize(img2,(img.shape[1],img.shape[0]))
        final = cv2.addWeighted(img2, 0.3, img, 0.7, 0) 

    hands = detector.findHands(final, False)
    # final = img
    cx, cy = 0,0
    detections = detectFaceOpenCVDnn(net, final)
    final, _ = process_detections(final, detections)

    if len(hands[0]) == 2:
        for i,hand in enumerate(hands[0]):
            if i == 0:
                cx, cy = hand['lmList'][9][:2]
            if i==1:
                    bbox = hand["bbox"]
                    handSize = bbox[2]
                    cx = cx + int((hand['lmList'][9][0] - cx)/2)
                    cy = cy + int(( hand['lmList'][9][1] - cy)/2)
                    final = Overlay(final, cake, cx, cy-100, 100)

    elif len(hands[0]) == 1:
        for i, hand in enumerate(hands):
            if i == 0:
                bbox = hand[0]["bbox"]
                handSize = bbox[2]
                cx, cy = hand[0]['lmList'][9][:2]
                final = Overlay(final, cake, cx, cy-100, 100)

    cv2.imshow("Crismas Cake", cv2.flip(final, 1))
    key = cv2.waitKey(2)
    if key == ord("Q") or key == ord('q') or key == 27:
        break 

cap.release()
cv2.destroyAllWindows()