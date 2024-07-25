import cv2
import mediapipe as mp
import numpy as np
# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    black_frame = np.zeros((frame.shape[0], frame.shape[1],3), dtype=np.uint8)

    if not success:
        break

    # Convert the frame color from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect facial landmarks
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                # Get the landmark coordinates
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                
                # Draw the keypoint
                # cv2.circle(frame, (x, y), radius=1, color=(0, 255, 0), thickness=-1)
                cv2.circle(black_frame,(x,y),radius=1,color=(14, 14, 232), thickness=-1)

    main_frame = np.concatenate((frame,black_frame),axis=1)

    # Display the frame with landmarks
    cv2.imshow('MediaPipe Face Mesh', main_frame)

    # Break the loop on 'q' key press
    key = cv2.waitKey(1)
    if (key == 27) or (key == ord("Q")) or (key == ord('q')):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
