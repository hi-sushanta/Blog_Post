
import os
import cv2
import datetime
import argparse
from ultralytics import YOLO
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8n-pose.pt', type=str, help='path to model weight')
    parser.add_argument('--sport', default='pushup', type=str,
                        help='Currently supported this type of exrecise "sit-up", "pushup" and "squat"')
    parser.add_argument('--input', default="0", type=str, help='path to input video')
    parser.add_argument('--save_dir', default=None, type=str, help='path to destination output')
    parser.add_argument('--show', default=True, type=bool, help='Display the result')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    model = YOLO(args.model)  

    # Open the camera or Video file
    if args.input.isnumeric():
        cap = cv2.VideoCapture(int(args.input))
    else: 
        cap = cv2.VideoCapture(args.input)
        

    # For working save result video
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

    reaching = False
    reaching_last = False
    state_keep = False
    counter = 0

    # Loop through the video frames
    while cap.isOpened():

        success, frame = cap.read()
        if success:

            plot_size_redio = max(frame.shape[1] / 960, frame.shape[0] / 540)

            results = model(frame)

            if results[0].keypoints.shape[1] == 0:
                if args.show:
                    put_text(frame, 'No Object', counter,
                             round(1000 / results[0].speed['inference'], 2), plot_size_redio)
                    scale = 640 / max(frame.shape[0], frame.shape[1]) 
                    show_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                    cv2.imshow("YOLOv8 Inference", show_frame)
                if args.save_dir is not None:
                    output.write(frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

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

            annotated_frame = plot(
                results[0], plot_size_redio,
            )
            put_text(
                annotated_frame, args.sport, counter, round(1000 / results[0].speed['inference'], 2), plot_size_redio)

            if args.show:
                scale = 640 / max(annotated_frame.shape[0], annotated_frame.shape[1]) 
                show_frame = cv2.resize(annotated_frame, (0, 0), fx=scale, fy=scale)
                cv2.imshow("YOLOv8 Inference", show_frame)

            if args.save_dir is not None:
                output.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    if args.save_dir is not None:
        output.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# Command to Follow : python3 AITrainer.py --sport <exercise_type> --model yolov8n-pose.pt --show True --input <path_to_your_video>
    
# Example For Push-Up: python AITrainer.py --sport pushup --model yolov8n-pose.pt --show True --input "pushup.mp4"  --save_dir "pushup_result"
# Example For Squat: python AITrainer.py --sport squat --model yolov8n-pose.pt --show True --input "squat.mp4" --save_dir "squat_result"
# Example For Sit-up: python AITrainer.py --sport sit-up --model yolov8n-pose.pt --show True --input "sit-up.mp4" --save_dir "sit-up-result"
