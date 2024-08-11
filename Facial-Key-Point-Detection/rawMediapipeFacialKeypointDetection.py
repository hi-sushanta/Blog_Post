import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)
window_name = "Raw MediaPipe"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)

output = cv2.VideoWriter("result.mp4", fourcc, fps, (1280, 480)) 
while cap.isOpened():
    success, frame = cap.read()
    black_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

    if not success:
        break

    # Convert the frame color from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect facial landmarks
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the face mesh on the original frame
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(165, 165, 168), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(123, 122, 128), thickness=1, circle_radius=1)
            )

            # Draw the face mesh on the black frame
            mp_drawing.draw_landmarks(
                image=black_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(165, 165, 168), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(123, 122, 128), thickness=1, circle_radius=1)
            )

    # Concatenate the original frame and the black frame side by side
    main_frame = np.concatenate((frame, black_frame), axis=1)
    output.write(main_frame)
    # Display the frame with landmarks
    cv2.imshow(window_name , main_frame)

    # Break the loop on 'q' key press
    key = cv2.waitKey(1)
    if (key == 27) or (key == ord("Q")) or (key == ord('q')):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
output.release()