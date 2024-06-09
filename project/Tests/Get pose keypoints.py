import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import LineString, Point

# Configurar la cámara Intel RealSense D435
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# Iniciar la transmisión
pipeline.start(config)

# Cargar el modelo YOLOv8 Pose
model = YOLO('yolov8n-pose.pt')  # Cambia esto al camino correcto de tu modelo YOLOv8 Pose

try:
    while True:
        # Capturar un frame de la cámara
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        # Convertir la imagen a un formato utilizable por OpenCV
        frame = np.asanyarray(color_frame.get_data())
        
        # Realizar la detección de poses
        results = model.predict(frame)

        # Dibujar los puntos de las articulaciones en la imagen
        for result in results:
            for i, r in enumerate(result.boxes.data.tolist()):
                keypoints = result.keypoints  
                if keypoints is not None and len(keypoints) > 0:
                    keypoints = keypoints[i].xy.cpu().numpy()  
                    for kp in keypoints:
                        if len(kp) >= 2: 
                            selected_values = [kp[6], kp[5], kp[12], kp[11]]
                            for values in selected_values:
                                x = int(values[0])
                                y = int(values[1])
                                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                            line1 = LineString([selected_values[0], selected_values[3]])   
                            line2 = LineString([selected_values[1], selected_values[2]])  
                            intersection = line1.intersection(line2) 
                            if intersection.is_empty:
                                print("Empty")
                            else: cv2.circle(frame, (int(intersection.x), int(intersection.y)), 5, (0, 255, 0), -1)
            
        # Mostrar la imagen con los puntos de las articulaciones
        cv2.imshow('YOLOv8 Pose Detection', frame)
        
        # Salir del loop con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Detener la transmisión
    pipeline.stop()
    cv2.destroyAllWindows()
