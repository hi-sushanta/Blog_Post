import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize OpenCV video capture
cap = cv2.VideoCapture("demovideo2.mp4")
window_name = "Pose Detection"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)

output = cv2.VideoWriter("FullBodyDetection.mp4", fourcc, fps, (2160 ,1920)) 
while cap.isOpened():
    success, frame = cap.read()
    # print(frame.shape[0],frame.shape[1])

    if not success:
        break
    black_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    # Convert the frame color from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect body landmarks
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        # Customize drawing style for original frame
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=4)
        )

        # Customize drawing style for black frame
        mp_drawing.draw_landmarks(
            image=black_frame,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=6, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=6, circle_radius=4)
        )

    # Concatenate the original frame and the black frame side by side
    main_frame = np.concatenate((frame, black_frame), axis=1)
    output.write(main_frame)
    # Display the frame with landmarks
    cv2.imshow(window_name, main_frame)

    # Break the loop on 'q' key press
    key = cv2.waitKey(1)
    if (key == 27) or (key == ord("Q")) or (key == ord('q')):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
output.release()

