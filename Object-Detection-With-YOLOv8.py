# demo video and output video here -- https://mega.nz/folder/ng0GkBZS#OcZbh_M8qA-i-nuw3RSG7g

from ultralytics import YOLO

YOLO_MODEL_PATH = "yolov8s.pt"

model = YOLO(YOLO_MODEL_PATH)

model.predict(source="demo_video.mp4", show=True, conf=0.50, save=False)

# OR

# model.predict(source='0',show=True,conf=0.50,save=True) acess webcam and detect object


