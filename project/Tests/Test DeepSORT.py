import cv2
import numpy as np
import random
import pyrealsense2 as rs
from ultralytics import YOLO
from tracker import Tracker
import cv2

# Configurar la cámara RealSense D435
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 60)
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
profile = pipeline.start(config)
tracker = Tracker()

# Cargar el modelo YOLOv8
model = YOLO("yolov8n-pose.pt")


try:
    while True:
        
        # Capturar un frame de la cámara RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Convertir de frame a matriz
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Aplicar mapa de color a la imagen de profundidad
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha = 0.5), cv2.COLORMAP_JET)
        
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        # Hacer inferencia con YOLOv8
        results = model(color_image)
        
        for result in results:
            detections = []
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, id = r
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                id = int(id)
                if score > 0.85:
                    detections.append([x1,y1,x2,y2,score])
            tracker.update(color_image, detections)
            for track in tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id
                if track_id is not None:
                    cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                    cv2.putText(color_image, "Id" + str(track_id), (int(x1), int(y1) + 20), cv2.FONT_HERSHEY_COMPLEX,0.7, (colors[track_id % len(colors)]),3)


                tracker.get_id(int(track_id))        


        cv2.imshow('Imagen', color_image)
        if cv2.waitKey(1) == ord('q'):
            break
        

finally:
    pipeline.stop()
