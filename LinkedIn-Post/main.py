import math
import cv2
from cvzone.HandTrackingModule import HandDetector

detector = HandDetector()


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
shield = cv2.imread('merry_crismas-removebg.png',flags=cv2.IMREAD_COLOR)


def mapFromTo(x,a,b,c,d):
    return (x-a)/(b-a)*(d-c)+c

def Overlay (background, overlay, x, y, size):
    background_h, background_w, c = background.shape
    imgScale = mapFromTo(size, 200, 20, 1.5, 0.2)
    overlay = cv2.resize(overlay, (0, 0), fx=imgScale, fy=imgScale)
    h, w, c = overlay.shape
    try:
        if x + w/2 >= background_w or y + h/2 >= background_h or x - w/2 <= 0 or y - h/2 <= 0:
            return background
        else:
            overlayImage = overlay[..., :3]
            mask = overlay / 255.0
            background[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = (1-mask)*background[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] + overlay
            return background
    except:
        return background

def findDistance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

showShield = True
changeTimer = 0

while True:
    success, img = cap.read()
    hands = detector.findHands(img, False)
    final = img
    if len(hands[0]) == 2:
        changeTimer += 1
        for i,hand in enumerate(hands[0]):
            if i==0 or i==1:
                if findDistance(hand["lmList"][4], hand["lmList"][4]) < 30:
                        bbox = hand["bbox"]
                        handSize = bbox[2]
                        cx, cy = hand["center"]
                        final = Overlay(img, shield, cx, cy, handSize)

    elif len(hands[0]) == 1:
        for i, hand in enumerate(hands):
            if i == 0:
                bbox = hand[0]["bbox"]
                handSize = bbox[2]
                cx, cy = hand[0]["center"]
                if 1 in detector.fingersUp(hand[0]):
                    final = Overlay(img, shield, cx, cy, handSize)

    cv2.imshow("Crismas Cake", cv2.flip(final, 1))
    key = cv2.waitKey(2)
    if key == ord("Q") or key == ord('q') or key == 27:
        break 

cap.release()
cv2.destroyAllWindows()