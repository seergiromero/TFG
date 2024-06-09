import os
import random

import cv2
from ultralytics import YOLO

from tracker import Tracker

# Informacion de donde esta el video y donde se quiere guardar
video_path = os.path.join('people.mp4')
video_out_path = os.path.join('.', 'deepsort.mp4')

# Obtener y analizar video
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

# Indicar el formato de video de salida
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),(frame.shape[1], frame.shape[0]))

# Inicializar modelo YOLO
model = YOLO("yolov8n-pose.pt")

# Inicializar DeepSORT
tracker = Tracker()

# Colores random a cada deteccion realizada para diferenciarlas
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

# Hacer el tracking de las personas del video hasta que este se acabe
while ret:

    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            id = int(id)
            if score > 0.85:
                detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
            cv2.putText(frame, "Id" + str(track_id), (int(x1), int(y1)),cv2.FONT_HERSHEY_COMPLEX, 0.7, (colors[track_id % len(colors)]), 2)
            

    cap_out.write(frame)
    ret, frame = cap.read()
    

cap.release()
cap_out.release()
