
import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from supervision.draw.utils import Point
from supervision.draw.color import Color
from supervision.draw.utils import draw_text
model = YOLO('yolov8x.pt')
heat_map_annotator = sv.HeatMapAnnotator()
video_path = "people_walking.mp4"
i = 0

import supervision as sv
from ultralytics import YOLO

model = YOLO('yolov8x.pt')

trace_annotator = sv.TraceAnnotator()
video_info = sv.VideoInfo.from_video_path(video_path=video_path)
frames_generator = sv.get_video_frames_generator(source_path=video_path)
tracker = sv.ByteTrack()
text_anchor = Point(x=600,y=20)
with sv.VideoSink(target_path="annotator.mp4", video_info=video_info) as sink:
   for i,frame in enumerate(frames_generator):
    # print(i)
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    # print(detections)
    detections = tracker.update_with_detections(detections)
    annotated_frame = trace_annotator.annotate(
        scene=frame,
        detections=detections)
    title = ""

    if i <= 31:
        title = "Dot-Annotator"
        dot_annotator = sv.DotAnnotator()
        annotated_frame = dot_annotator.annotate(
             scene=annotated_frame,
             detections=detections
         )
    elif (i > 31) and (i <= 31*2):
        title = "BoundingBox-Annotator"
        bounding_box_annotator = sv.BoundingBoxAnnotator()
        annotated_frame = bounding_box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
        )
    elif (i > 31*2) and (i <= 31*3):
        title = "BoxCorner-Annotator"
        corner_annotator = sv.BoxCornerAnnotator()
        annotated_frame = corner_annotator.annotate(
        scene=frame.copy(),
        detections=detections
        )
    elif (i > 31*3) and (i <= 31*4):
        title = "Color-Annotator"
        color_annotator = sv.ColorAnnotator()
        annotated_frame = color_annotator.annotate(
        scene=frame.copy(),
        detections=detections
        )
    elif (i > 31*4) and (i <= 31*5):
        title = "Circle-Annotator"
        circle_annotator = sv.CircleAnnotator()
        annotated_frame = circle_annotator.annotate(
        scene=frame.copy(),
        detections=detections
        )
    elif (i > 31*5) and (i <= 31*6):
        title = "Triangle-Annotator"
        triangle_annotator = sv.TriangleAnnotator()
        annotated_frame = triangle_annotator.annotate(
        scene=frame.copy(),
        detections=detections
        )
    elif (i > 31*6) and (i <= 31*7):
        title = "Ellipse-Annotator"
        ellipse_annotator = sv.EllipseAnnotator()
        annotated_frame = ellipse_annotator.annotate(
        scene=frame.copy(),
        detections=detections
        )
    elif (i > 31*7) and (i <= 31*8):
        title = "HaloAnnotator"
        halo_annotator = sv.HaloAnnotator()
        annotated_frame = halo_annotator.annotate(
        scene=frame.copy(),
        detections=detections
        )
    elif (i > 31*8) and (i <= 31*9):
        title = "HeatMap-Annotator"
        heat_map_annotator = sv.HeatMapAnnotator()

        annotated_frame = heat_map_annotator.annotate(
                    scene=frame.copy(),
                    detections=detections)
    elif (i > 31*9) and (i <= 31*9):
        title = "MaskAnnotator"
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(
            scene=frame.copy(),
            detections=detections
            )
    elif (i > 31*10):
        title = "Polygon-Annotator"
        polygon_annotator = sv.PolygonAnnotator()
        annotated_frame = polygon_annotator.annotate(
        scene=frame.copy(),
         detections=detections
        )

    labels = [
            result.names[class_id]
            for class_id
            in detections.class_id
            ]

    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame,
                        detections=detections,
                        labels=labels
                        )
     
    annotated_frame = draw_text(annotated_frame,f"Trace-Label-{title}",
              text_anchor,Color(r=148, g=0, b=255),1,background_color=Color(r=243, g=248, b=255))

    sink.write_frame(frame=annotated_frame)