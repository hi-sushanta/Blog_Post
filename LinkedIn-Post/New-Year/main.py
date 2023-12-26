import math
import cv2
from cvzone.HandTrackingModule import HandDetector 
import numpy as np

# # Initialize cascaded classifier class.
# face_cascade = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")


# detector = HandDetector(maxHands=1)

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)


# snow_cap = cv2.VideoCapture("first.mp4")
# snow_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# snow_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

# # Function for detecting faces in and image
# def detectFaceOpenCVDnn(net, frame):
#     # create a blot from image and some pre-processing
#     blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
#     # Set the blob input to the model
#     net.setInput(blob)
#     detections = net.forward()
#     return detections

# # # Function for annotating the image with bounding boxes for each detected face.
# # def process_detections(frame,, detections, conf_threshold=0.5):
# #     bboxes = []
# #     frame_h = frame.shape[0]
# #     frame_w = frame.shape[1]
# #     # Loop over all detection and draw bounding boxes around each faces
# #     for i in range(detections.shape[2]):
# #         confidence = detections[0, 0, i, 2]
# #         if confidence > conf_threshold:
# #             x1 = int(detections[0, 0, i, 3] * frame_w)
# #             y1 = int(detections[0, 0, i, 4] * frame_h)
# #             x2 = int(detections[0, 0, i, 5] * frame_w)
# #             y2 = int(detections[0, 0, i, 6] * frame_h)
# #             bboxes.append([x1, y1, x2, y2])
# #             c1 = int(x1 + (x2 - x1)/2)
# #             c2 = int(y1 + (y2 - y1)/2)
# #             x3 = c1
# #             y3 = c2 - (c2 - y1)
# #             x4 = c1 
# #             y4 = c2 + (y2 - c2)
           
# #             frame = Overlay(frame, santa_hat, x3-15,y3-15, 130)
# #             frame = Overlay(frame,santa_beard,x4,y4-40,90)

#     return frame, bboxes
# # Function to load the DNN model.
# def load_model():
#     modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
#     configFile = "deploy.prototxt"
#     net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
#     return net

# def mapFromTo(x,a,b,c,d):
#     return (x-a)/(b-a)*(d-c)+c

# def Overlay (background, overlay, x, y, size):
#     background_h, background_w, c = background.shape
#     imgScale = mapFromTo(size, 200, 40, 1.3, 0.3)
#     overlay = cv2.resize(overlay, (0, 0), fx=imgScale, fy=imgScale)
#     h, w, c = overlay.shape
#     try:
#         if x + w/2 >= background_w or y + h/2 >= background_h or x - w/2 <= 0 or y - h/2 <= 0:
#             return background
#         else:
#             overlayImage = overlay[..., :3]
#             mask = (overlay / 255.0)
#             background[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = (1-mask)*background[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] + overlayImage
#             return background
#     except:
#         return background



# net = load_model()
# while True:
#     success, img = cap.read()
    
#     landmarks = detector.findHands(img,False)
#     fingerCount = 0
#     print(landmarks)
#     if len(landmarks[0]) == 1:
#         for landmark in landmarks:
#             if landmark.type == "finger" and landmark.bend < 30:
#                 fingerCount += 1
        
#         if fingerCount == 1:
#             success2,img2 = snow_cap.read()
#             if success2:
#                 img2 = cv2.resize(img2,(img.shape[1],img.shape[0]))
#                 final = cv2.addWeighted(img2, 0.3, img, 0.7, 0) 
#             else:
#                 snow_cap = cv2.VideoCapture('first.mp4')
#                 success2, img2 = snow_cap.read()
#                 img2 = cv2.resize(img2, (img.shape[1],img.shape[0]))
#                 final = cv2.addWeighted(img2,0.3,img,0.7,0)
#         elif fingerCount == 2:
#             final = img
#         elif fingerCount == 3:
#             final = img
#     else:
#         final = img

#     # elif success2 == False:
#     #     snow_cap = cv2.VideoCapture('snow-fall.mp4')
#     #     success2,img2 = snow_cap.read()
#     #     img2 = cv2.resize(img2,(img.shape[1],img.shape[0]))
#     #     final = cv2.addWeighted(img2, 0.3, img, 0.7, 0) 

#     # hands = detector.findHands(final, False)
#     # # final = img
#     # cx, cy = 0,0
#     # detections = detectFaceOpenCVDnn(net, final)

#     # if len(hands[0]) == 2:
#     #     for i,hand in enumerate(hands[0]):
#     #         if i == 0:
#     #             cx, cy = hand['lmList'][9][:2]
#     #         if i==1:
#     #                 bbox = hand["bbox"]
#     #                 handSize = bbox[2]
#     #                 cx = cx + int((hand['lmList'][9][0] - cx)/2)
#     #                 cy = cy + int(( hand['lmList'][9][1] - cy)/2)
#     #                 final = Overlay(final, cake, cx, cy-100, 100)

#     # elif len(hands[0]) == 1:
#     #     for i, hand in enumerate(hands):
#     #         if i == 0:
#     #             bbox = hand[0]["bbox"]
#     #             handSize = bbox[2]
#     #             cx, cy = hand[0]['lmList'][9][:2]
#     #             final = Overlay(final, cake, cx, cy-100, 100)

#     cv2.imshow("Crismas Cake", cv2.flip(final, 1))
#     key = cv2.waitKey(2)
#     if key == ord("Q") or key == ord('q') or key == 27:
#         break 

# cap.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)


snow_cap = cv2.VideoCapture("first.mp4")
snow_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
snow_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Initially set finger count to 0 for each cap
    fingerCount = 0
    final = image.copy()
    if results.multi_hand_landmarks:

      for hand_landmarks in results.multi_hand_landmarks:
        # Get hand index to check label (left or right)
        handIndex = results.multi_hand_landmarks.index(hand_landmarks)
        handLabel = results.multi_handedness[handIndex].classification[0].label

        # Set variable to keep landmarks positions (x and y)
        handLandmarks = []

        # Fill list with x and y positions of each landmark
        for landmarks in hand_landmarks.landmark:
          handLandmarks.append([landmarks.x, landmarks.y])

        # Test conditions for each finger: Count is increased if finger is 
        #   considered raised.
        # Thumb: TIP x position must be greater or lower than IP x position, 
        #   deppeding on hand label.
        if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
          fingerCount = fingerCount+1
        elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
          fingerCount = fingerCount+1

        # Other fingers: TIP y position must be lower than PIP y position, 
        #   as image origin is in the upper left corner.
        if handLandmarks[8][1] < handLandmarks[6][1]:       #Index finger
          fingerCount = fingerCount+1
        if handLandmarks[12][1] < handLandmarks[10][1]:     #Middle finger
          fingerCount = fingerCount+1
        if handLandmarks[16][1] < handLandmarks[14][1]:     #Ring finger
          fingerCount = fingerCount+1
        if handLandmarks[20][1] < handLandmarks[18][1]:     #Pinky
          fingerCount = fingerCount+1
        
        if fingerCount == 1:
            success2,img2 = snow_cap.read()
            if success2:
                img2 = cv2.resize(img2,(image.shape[1],image.shape[0]))
                final = cv2.addWeighted(img2, 0.3, image, 0.7, 0) 
            else:
                snow_cap = cv2.VideoCapture('first.mp4')
                success2, img2 = snow_cap.read()
                img2 = cv2.resize(img2, (image.shape[1],image.shape[0]))
                final = cv2.addWeighted(img2,0.3,image,0.7,0)
        

        # Draw hand landmarks 
        mp_drawing.draw_landmarks(
            final,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    # Display finger count
    cv2.putText(final, str(fingerCount), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

    # Display image
    cv2.imshow('MediaPipe Hands', cv2.flip(final,1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()