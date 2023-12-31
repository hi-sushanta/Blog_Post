import cv2
import HandTrackingModule as htm
import math

wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

video_path = "first.mp4"
snow_cap = cv2.VideoCapture(video_path)
snow_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
snow_cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)

snow_cap2 = cv2.VideoCapture("second.mp4")
snow_cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
snow_cap2.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)

snow_cap3 = cv2.VideoCapture("third.mp4")
snow_cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
snow_cap3.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)


detector = htm.handDetector(detectionCon=1)


step = 0
attach = 0
i = 0

def read_image(fcap,img,video_path,item=1):
    success2,img2 = fcap.read()
    if success2:
            img2 = cv2.resize(img2,(img.shape[1],img.shape[0]))
            img = cv2.addWeighted(img2, 0.4, img, 0.6, 0.0) 
    elif (success2 == False) and (item == 1):
            global snow_cap
            snow_cap = cv2.VideoCapture(video_path)
            success2,img2 = snow_cap.read()
            img2 = cv2.resize(img2,(img.shape[1],img.shape[0]))
            img = cv2.addWeighted(img, 0.4, img2, 0.6, 0)
    elif (success2 == False) and (item == 2):
            global snow_cap2
            snow_cap2 = cv2.VideoCapture(video_path)
            success2,img2 = snow_cap2.read()
            img2 = cv2.resize(img2,(img.shape[1],img.shape[0]))
            img = cv2.addWeighted(img, 1, img2, 1, 0)
    elif (success2 == False) and (item == 3):
         global snow_cap3
         snow_cap3 = cv2.VideoCapture(video_path)
         success2, img2 = snow_cap3.read()
         img2 = cv2.resize(img2,(img.shape[1],img.shape[0]))
         img = cv2.addWeighted(img, 1, img2, 1,0)
    elif (success2 == False) and (item == 4):
         global snow_cap4
         snow_cap4 = cv2.VideoCapture(video_path)
         success2, img2 = snow_cap4.read()
         img2 = cv2.resize(img2, (img.shape[1],img.shape[0]))
         img = cv2.addWeighted(img2, 0.3, img ,0.7, 0)
    elif (success2 == False) and (item == 5):
         global snow_cap5
         snow_cap5 = cv2.VideoCapture(video_path)
         success2, img2 = snow_cap5.read()
         img2 = cv2.resize(img2, (img.shape[1],img.shape[0]))
         img = cv2.addWeighted(img2, 0.3, img, 0.7,0)

    return img 

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList[0]) != 0:
        x1, y1 = lmList[0][4][1], lmList[0][4][2]
        x2, y2 = lmList[0][12][1], lmList[0][12][2]
        x3, y3 = lmList[0][8][1], lmList[0][8][1]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cx2,cy2 = (x1+x3)//2, (y1+y3)//2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        # length2 = math.hypot(x3 - x1,y3 - y1)
        length2 = math.hypot(x1-x3,y1 - y3)

        

        if length < 20:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
            attach = 1
        elif length2 > 150:
            cv2.circle(img,(x1,y1),16,(255,0,255),cv2.FILLED)
            cv2.circle(img,(x3,y3),15,(255,0,255),cv2.FILLED)
            cv2.line(img,(x1,y1),(x3,y3),(255,0,255),3)
            cv2.circle(img,(cx2,cy2),16,(255,0,255),cv2.FILLED)
            if attach == 1:
                attach = 0
                step += 1
                    
    count = "0"
    img = cv2.flip(img,1)

    if step == 1:
        img = read_image(snow_cap,img,video_path)
        count = "1"
    elif step == 2:
        video_path = "second.mp4"
        img = read_image(snow_cap2,img,video_path,2)
        count = "2" 
    elif step == 3:
         video_path = "third.mp4"
         img = read_image(snow_cap3,img, video_path,3)
         count = "3"
    if count != "0":
         cv2.putText(img,count,(img.shape[1]-100,img.shape[0] - (img.shape[0]-100)),3,2,(197, 197, 255),4,cv2.LINE_AA)

    cv2.imshow("Img", img)
    key = cv2.waitKey(1)
    if key == ord("q") or key == ord("Q") or key == 27:
        break 

snow_cap.release()
snow_cap2.release()
snow_cap3.release()
cv2.destroyAllWindows()