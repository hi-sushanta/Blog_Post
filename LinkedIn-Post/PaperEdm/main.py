import cv2 
import numpy as np
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
# Create a new window
cv2.namedWindow("hello", cv2.WINDOW_NORMAL)

# # Set the window's property to full screen
cv2.setWindowProperty("hello", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

img = cv2.imread(r"C:\Users\hi_ai\Documents\MyProject\Blog_Post\LinkedIn-Post\PaperEdm\piano.png")
img = cv2.resize(img,(640,170))

detector = htm.handDetector(detectionCon=1)


while True:
    is_frame, frame = cap.read()
    frame = cv2.flip(frame,1)
    if not is_frame:
        break

    frame = detector.findHands(frame)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList[0]) != 0:
        x1, y1 = lmList[0][4][1], lmList[0][4][2]
        x2, y2 = lmList[0][12][1], lmList[0][12][2]
        x3, y3 = lmList[0][8][1], lmList[0][8][1]

        # cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        # cv2.circle(frame, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        # cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        # cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)


    roi = frame[250:420,0:640]
    dst = cv2.addWeighted(roi,0.1,img,0.7,0)
    frame[250:420,0:640] = dst

    cv2.imshow("hello",frame)
    # cv2.imshow("why",img)

    key = cv2.waitKey(1)

    if (key == ord("Q")) or (key == ord("q")) or (key == 27):
        break

cap.release()
cv2.destroyAllWindows()