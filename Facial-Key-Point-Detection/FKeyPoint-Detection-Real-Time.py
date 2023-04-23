import cv2
import torch
import numpy as np
from torchvision import transforms
from Model import DFKModel

face_cascade = cv2.CascadeClassifier(
    r"C:\Users\hiwhy\OneDrive\Documents\Blog_post\Facial-Key-Point-Detection\haarcascade_frontalface_default.xml")

model = DFKModel()
model.load_state_dict(torch.load('my_model.pth', map_location=torch.device('cpu')))
model = model.to(torch.device('cpu'))
cap = cv2.VideoCapture(0)
window_name = 'Facial Keypoint Detection'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)

transform = transforms.Compose([
    transforms.ToTensor()])

while True:
    is_frame, frame = cap.read()
    if not is_frame:
        break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.25, 6)
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face_crop = gray_image[y:y + h, x:x + w]
            resized = cv2.resize(face_crop, (96, 96)) / 255
            normalised_image = np.copy(resized)
            reshaped_image = normalised_image.reshape(-1, 96, 96, 1)
            key_image = transforms.ToTensor()(reshaped_image[0])
            with torch.no_grad():
                model.eval()
                facial_keypoint_predictions = model(key_image.unsqueeze(0).type(torch.float32))
            output = facial_keypoint_predictions.squeeze().cpu().numpy() * 255 + 255
            label_p = output.reshape(-1, 2)

            for i in range(label_p.shape[0]):
                cv2.circle(frame, (int((label_p[i, 0]) * (w / 96) + x), int((label_p[i, 1]) * (h / 96) + y)), 4,
                        (0, 255, 0), -1)
    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1)

    if key == ord("q") or key == ord("Q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
