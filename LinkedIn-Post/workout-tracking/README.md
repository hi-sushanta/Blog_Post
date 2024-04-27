# A Push-Up counter that uses cutting-edge tech!

Using YOLOv8 (fancy AI stuff), I built a model that can track your push-ups and count them super accurately. No more losing track or relying on guesswork!

Hope you're as excited as I am! Let me know what you think!

***These steps follow when are you excited to use this project on your local machine:***

1. The first step is to clone this project on your local machine

2. The second step is to go to the specific workout tracking project directory

3. The third step is to choose one of those commands based on the exercise:

```
# Command to Follow : python3 AITrainer.py --sport <exercise_type> --model yolov8n-pose.pt --show True --input <path_to_your_video>
    
# Example For Push-Up: python AITrainer.py --sport pushup --model yolov8n-pose.pt --show True --input "pushup.mp4"  --save_dir "pushup_result"
# Example For Squat: python AITrainer.py --sport squat --model yolov8n-pose.pt --show True --input "squat.mp4" --save_dir "squat_result"
# Example For Sit-up: python AITrainer.py --sport sit-up --model yolov8n-pose.pt --show True --input "sit-up.mp4" --save_dir "sit-up-result"
```
