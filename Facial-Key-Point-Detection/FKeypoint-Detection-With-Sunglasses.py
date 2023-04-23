import cv2
import torch
import numpy as np
from torchvision import transforms
from Model import DFKModel

face_cascade = cv2.CascadeClassifier(
    r"C:\Users\hiwhy\OneDrive\Documents\Blog_post\Facial-Key-Point-Detection\haarcascade_frontalface_default.xml")

sunglass_img = cv2.imread('sunglasses2.jpg', cv2.IMREAD_UNCHANGED)
sunglass_img = cv2.cvtColor(sunglass_img, cv2.COLOR_BGRA2RGBA)
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
            f_key_point_w = int(label_p[2][0] * (w / 96) + x)
            f_key_point_h = int(label_p[2][1] * (h / 96) + y)
            f2_key_point_w = int(label_p[4][0] * (w / 96) + x)
            f2_key_point_h = int(label_p[4][1] * (h / 96) + y)

            point1 = int((f_key_point_w + f2_key_point_w)/2) # * (w/96) + x)
            point2 = int((f_key_point_h + f2_key_point_h)/2) # * (h/96) + y)
            roi = frame[point2 - 50:point2 + 50, point1 - 100:point1 + 100]
            # Create a mask from the sunglasses image alpha channel
            sunglass_mask = sunglass_img[..., :3][..., ::-1]
            # sunglass_mask = np.zeros_like(sunglass_img[..., :3])
            # sunglass_mask[..., 2] = 255

            sunglass_mask = cv2.resize(sunglass_mask,(200,100))
            # Change the color of the sunglasses to black
            # sunglass_mask[sunglass_mask[:,:,0] == 0] = [48,48,48]
            # sunglass_mask[sunglass_mask[:,:,1] == 0] = [48,48,48]
            # sunglass_mask[sunglass_mask[:,:,2] == 0] = [48,48,48]
            # sunglass_mask[sunglass_mask[:, :, 0] >= 150] = [0, 0, 0]
            # sunglass_mask[sunglass_mask[:, :, 1] >= 150] = [0, 0, 0]
            # sunglass_mask[sunglass_mask[:, :, 2] >= 150] = [0, 0, 0]
            # sunglass_mask[sunglass_mask[:, :, 0] == 48] = [255, 255, 255]
            # sunglass_mask[sunglass_mask[:, :, 1] == 48] = [255,255,255]
            # sunglass_mask[sunglass_mask[:, :, 2] == 48] = [255, 255, 255]

            print(roi.shape)
            print(sunglass_mask.shape)
            img = cv2.addWeighted(sunglass_mask, 1, roi, 1, 0)
            # img = cv2.addWeighted(sunglass_mask, 1, roi, 1, 0, dtype=cv2.CV_8UC3)
            frame[point2 - 50:point2 + 50, point1 - 100:point1 + 100] = img


            # for i in range(label_p.shape[0]):
            #     cv2.circle(frame, (int((label_p[i, 0]) * (w / 96) + x), int((label_p[i, 1]) * (h / 96) + y)), 4,
            #             (0, 255, 0), -1)
            #     cv2.circle(frame,(f_key_point_w,f_key_point_h),4,(255,0,0),-1)
            #     cv2.circle(frame,(f2_key_point_w,f2_key_point_h),4,(255,255,255),-1)
            #     cv2.circle(frame,(point1,point2),4,(255,0,255),-1)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1)

    if key == ord("q") or key == ord("Q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
