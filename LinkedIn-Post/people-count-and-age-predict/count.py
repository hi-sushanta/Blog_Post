
import cv2
import supervision as sv
from ultralytics import YOLO
import dlib

model = YOLO("yolov8n.pt")
corner_annotator = sv.BoxCornerAnnotator()


def fannotate(frame,detections):
    annotated_frame = corner_annotator.annotate(
        scene=frame.copy(),
        detections=detections
        )
    return annotated_frame

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)

# Age Prediction Model:
# ------------ Model for Age detection --------# 
age_weights = "age_deploy.prototxt"
age_config = "age_net.caffemodel"
age_Net = cv2.dnn.readNet(age_config, age_weights) 
  
# Model requirements for image 
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
           '(25-32)', '(38-43)', '(48-53)', '(60-100)'] 
model_mean = (78.4263377603, 87.7689143744, 114.895847746) 

  
# ------------- Model for face detection---------# 
face_detector = dlib.get_frontal_face_detector() 

while True:
    isFrame, frame = cap.read()
    if not isFrame:
        break 
    cframe = frame.copy()
    Boxes = []  # to store the face co-ordinates 
    mssg = 'Face Detected'  # to display on image 
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    labels = [
        model.model.names[class_id]  if (class_id == 0) and ((confidence*100.0) >= 43.0) else None
        for class_id,confidence
        in zip(detections.class_id,detections.confidence)
    ]
    labels = list(filter(lambda x: x is not None, labels))
    frame = fannotate(frame,detections)
    h,w = frame.shape[:2]
    cv2.putText(frame,f"{len(labels)}",(int(w/8)+30,h-50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv2.LINE_AA)
    cv2.putText(frame,"Pepole",(int(w/8),h-20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv2.LINE_AA)
     
    # Age Prediction
    # converting BGR to RGB and Then convert RGB to grayscale 
    img_rgb = cv2.cvtColor(cframe,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) 
  
    # -------------detecting the faces--------------# 
    faces = face_detector(img_gray) 
    age = " "
    if not faces:
        age = " "
    else:
        for face in faces: 
            x = face.left()  # extracting the face coordinates 
            y = face.top() 
            x2 = face.right() 
            y2 = face.bottom() 
    
            # rescaling those coordinates for our image 
            box = [x, y, x2, y2] 
            Boxes.append(box) 
        for box in Boxes: 
            # print(f"Box0:{box[0]},Box1:{box[1]},Box2:{box[2]},Box3:{box[3]}")
            face = frame[box[1]:box[3], box[0]:box[2]] 
    
            # ----- Image preprocessing --------# 
            blob = cv2.dnn.blobFromImage( 
                face, 1.0, (227, 227), model_mean, swapRB=False) 
    
            # -------Age Prediction---------# 
            age_Net.setInput(blob) 
            age_preds = age_Net.forward() 
            age = f"{ageList[age_preds[0].argmax()]}" 
            cv2.putText(frame,f"{age}",(box[0],box[1]-70),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv2.LINE_AA)
    

    cv2.imshow("Age With Person",frame)
    key = cv2.waitKey(1)
    if key == ord("Q") or key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
