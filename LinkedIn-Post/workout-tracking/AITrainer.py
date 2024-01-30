
import os
import cv2
import datetime
import argparse
from ultralytics import YOLO
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8s-pose.pt', type=str, help='path to model weight')
    parser.add_argument('--sport', default='squat', type=str,
                        help='Currently supported "sit-up", "pushup" and "squat"')
    parser.add_argument('--input', default="0", type=str, help='path to input video')
    parser.add_argument('--save_dir', default=None, type=str, help='path to save output')
    parser.add_argument('--show', default=True, type=bool, help='show the result')
    args = parser.parse_args()
    return args


def main():
    # Obtain relevant parameters
    args = parse_args()
    # Load the YOLOv8 model
    model = YOLO(args.model)  

    # Open the video file or camera
    if args.input.isnumeric():
        cap = cv2.VideoCapture(int(args.input))
    else: 
        cap = cv2.VideoCapture(args.input)
        
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    # For save result video
    if args.save_dir is not None:
        save_dir = os.path.join(
            args.save_dir, args.sport,
            datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        output = cv2.VideoWriter(os.path.join(save_dir, 'result.mp4'), fourcc, fps, size)

    # Set variables to record motion status
    reaching = False
    reaching_last = False
    state_keep = False
    counter = 0

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Set plot size redio for inputs with different resolutions
            plot_size_redio = max(frame.shape[1] / 960, frame.shape[0] / 540)

            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Preventing errors caused by special scenarios
            if results[0].keypoints.shape[1] == 0:
                if args.show:
                    put_text(frame, 'No Object', counter,
                             round(1000 / results[0].speed['inference'], 2), plot_size_redio)
                    scale = 640 / max(frame.shape[0], frame.shape[1]) 
                    show_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                    cv2.imshow("YOLOv8 Inference", show_frame)
                if args.save_dir is not None:
                    output.write(frame)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # Get hyperparameters
            left_points_idx = sport_list[args.sport]['left_points_idx']
            right_points_idx = sport_list[args.sport]['right_points_idx']

            # Calculate angle
            angle = calculate_angle(results[0].keypoints, left_points_idx, right_points_idx)

            # Determine whether to complete once
            if angle < sport_list[args.sport]['maintaining']:
                reaching = True
            if angle > sport_list[args.sport]['relaxing']:
                reaching = False

            if reaching != reaching_last:
                reaching_last = reaching
                if reaching:
                    state_keep = True
                if not reaching and state_keep:
                    counter += 1
                    state_keep = False

            # Visualize the results on the frame
            annotated_frame = plot(
                results[0], plot_size_redio,
                # sport_list[args.sport]['concerned_key_points_idx'],
                # sport_list[args.sport]['concerned_skeletons_idx']
            )
            # annotated_frame = results[0].plot(boxes=False)

            # add relevant information to frame
            put_text(
                annotated_frame, args.sport, counter, round(1000 / results[0].speed['inference'], 2), plot_size_redio)

            # Display the annotated frame
            if args.show:
                scale = 640 / max(annotated_frame.shape[0], annotated_frame.shape[1]) 
                show_frame = cv2.resize(annotated_frame, (0, 0), fx=scale, fy=scale)
                cv2.imshow("YOLOv8 Inference", show_frame)

            if args.save_dir is not None:
                output.write(annotated_frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    if args.save_dir is not None:
        output.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# Command to RUN : python3 AITrainer.py --sport <exercise_type> --model yolov8s-pose.pt --show True --input <path_to_your_video>
# Example: python AITrainer.py --sport pushup --model yolov8s-pose.pt --show True --input "C:\Users\hi_ai\Downloads\pushup3.mp4"  --save_dir "C:\Users\hi_ai\Downloads\pushup_result\"