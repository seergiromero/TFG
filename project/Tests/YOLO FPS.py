import cv2
import numpy as np
import random
import pyrealsense2 as rs
from ultralytics import YOLO
import time
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Configurar la cámara RealSense D435
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
config.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 60)
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
pipeline.start(config)

# Cargar el modelo YOLOv8
model = YOLO("yolov8n-pose.pt")

# Variables para medir los FPS de la cámara
fps_interval = 10  # Intervalo de tiempo para calcular los FPS
start_time = time.time()
frame_count = 0

# Listas para almacenar los datos de FPS de la cámara y la velocidad de procesamiento
camera_fps_list = []
processing_speed_list = []

try:
    while True:
        # Capturar un frame de la cámara RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,alpha=0.5), cv2.COLORMAP_JET)

        # Hacer inferencia con YOLOv8
        start_inference_time = time.time()  # Tiempo de inicio de la inferencia
        results = model(color_image)
        end_inference_time = time.time()  # Tiempo de finalización de la inferencia

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
                    detections.append([x1, y1, x2, y2, score])

                cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)),(colors[id % len(colors)]), 3)

        cv2.imshow('Imagen', color_image)
        if cv2.waitKey(1) == ord('q'):
            break

        # Calcular FPS de la cámara y velocidad de procesamiento
        frame_count += 1
        if frame_count % fps_interval == 0:
            end_time = time.time()
            fps = fps_interval / (end_time - start_time)
            inference_time = (end_inference_time - start_inference_time) / fps_interval
            
            # Almacenar los datos en las listas
            camera_fps_list.append(fps)
            processing_speed_list.append(1 / inference_time)
            
            # Imprimir los valores de FPS de la cámara y velocidad de procesamiento
            print(f"FPS de la cámara: {fps:.2f}, Velocidad de procesamiento: {1 / inference_time:.2f} FPS")
            
            start_time = time.time()

finally:
    pipeline.stop()

    # Crear plots al finalizar la ejecución
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(camera_fps_list, label='FPS de la cámara')
    plt.xlabel('Tiempo')
    plt.ylabel('FPS')
    plt.title('FPS de la cámara')

    plt.subplot(2, 1, 2)
    plt.plot(processing_speed_list, label='Velocidad de procesamiento (FPS)')
    plt.xlabel('Tiempo')
    plt.ylabel('FPS')
    plt.title('Velocidad de procesamiento')

    plt.tight_layout()
    plt.show()
