import cv2
import numpy as np
import os
import HandTrackingModule as htm



folderPath = "Drawing-Bar"

myList = os.listdir(folderPath)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0] # Default header
drawColor = (127, 0, 255) # Default Color 

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Display Full screen
win_name = "Virtual-Frame"
cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(win_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("VirtualPenDemo.mp4",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))
# Maximum one hand supported.
detector = htm.handDetector(detectionCon=1,maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

brushThickness = 6
eraserThickness = 100

while True:

    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_copy = img.copy()
    img = detector.findHands(img)
    lmList ,bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:


        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0

            if y1 < 80:
                if 80 < x1 < 180:
                    header = overlayList[0]
                    drawColor = (127, 0, 255)
                elif 220 < x1 < 300:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 360 < x1 < 420:
                    header = overlayList[2]
                    drawColor = (0, 222, 255)
                elif 460 < x1 < 540:
                    header = overlayList[3]
                    drawColor = (0, 255, 0)
                elif 600 < x1 < 680:
                    header = overlayList[4]
                    drawColor = (0, 128, 255)
                elif 740 < x1 < 820:
                    header = overlayList[5]
                    drawColor = (0,0,255)
                elif 880 < x1 < 980:
                    header = overlayList[6]
                    drawColor = (255,0,255)
                elif 1050 < x1 < 1200:
                    header = overlayList[7]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED,lineType=cv2.LINE_AA)
            

        if fingers[1] and fingers[2] == False:
            if xp == 0 and yp == 0:
                xp, yp = x1, y1


            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness,lineType=cv2.LINE_AA)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness,lineType=cv2.LINE_AA)
            
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness,lineType=cv2.LINE_AA)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness,lineType=cv2.LINE_AA)

            xp, yp = x1, y1
        
        # If finger gesture is detected Thumb_Up then clear my all text.
        finger_ok = detector.all_clear(img_copy)

        if finger_ok is not None:
            print(finger_ok)
            if finger_ok == "Thumb_Up":
                imgCanvas = np.zeros((720, 1280, 3), np.uint8)



    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    img[0:80, 0:1280] = header
    cv2.imshow(win_name, img)
    video_writer.write(img)

    
    key = cv2.waitKey(1)

    if (key == 27) or (key == ord("Q")) or (key == ord("q")):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()