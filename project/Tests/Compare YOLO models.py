import cv2
import numpy as np
import random
import pyrealsense2 as rs
from ultralytics import YOLO
import time
import cv2
import matplotlib.pyplot as plt

# Configurar la cámara RealSense D435
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
config.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 60)
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
pipeline.start(config)

# Cargar el modelo YOLOv8
model = YOLO("yolov8s-pose.pt")

# Variables para medir los FPS de la cámara
fps_interval = 10  # Intervalo de tiempo para calcular los FPS
start_time = time.time()
frame_count = 0
fps_list = []

# Listas para almacenar los datos de velocidad
preprocess_speed_list = []
inference_speed_list = []
postprocess_speed_list = []

# Lista para almacenar el número de personas detectadas
people_count_list = []


try:
    while True:
        # Capturar un frame de la cámara RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Crear imagen con los datos de los frames
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Aplicar mapa de color a la imagen de profundidad
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,alpha=0.5), cv2.COLORMAP_JET)
        
        # Uso de YOLOv8 en la imagen
        results = model(color_image)

        # Analizar la información de velocidad
        for result in results:

            #Velcidad de procesamiendo de YOLO
            speed_info = result.speed
            preprocess_speed = speed_info['preprocess']
            inference_speed = speed_info['inference']
            postprocess_speed = speed_info['postprocess']

            #Poner las velocidades es su base de datos correspondiente    
            preprocess_speed_list.append(preprocess_speed)
            inference_speed_list.append(inference_speed)
            postprocess_speed_list.append(postprocess_speed)

            print(f"Speed: {preprocess_speed}ms preprocess, {inference_speed}ms inference, {postprocess_speed}ms postprocess")
        
        # Calcular FPS de la cámara
        frame_count += 1
        if frame_count % fps_interval == 0:
            end_time = time.time()
            fps = fps_interval / (end_time - start_time)
            fps_list.append(fps)
            print(f"FPS de la cámara: {fps:.2f}")
            start_time = time.time()

        # Variable para monitorear numero de personas detectadas
        num_people_detected = 0

        # Realizar detecciones y dibujar rectángulos
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
                    num_people_detected += 1
                cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)),(colors[id % len(colors)]), 3)

        # Actualizar base de datos personas detectadas en este instante
        people_count_list.append(num_people_detected)


        cv2.imshow('Imagen', color_image)
        if cv2.waitKey(1) == ord('q'):
            break



finally:
 
    # Calcular la media de los valores
    preprocess_speed_mean = np.mean(preprocess_speed_list)
    inference_speed_mean = np.mean(inference_speed_list)
    postprocess_speed_mean = np.mean(postprocess_speed_list)
    fps_mean = np.mean(fps_list)

    print(f"Media del preprocesamiento: {preprocess_speed_mean} ms")
    print(f"Media de la inferencia: {inference_speed_mean} ms")
    print(f"Media del postprocesamiento: {postprocess_speed_mean} ms")
    print(f"Media de los FPS: {fps_mean:.2f}")

    # Crear plots al finalizar la ejecución
    plt.figure(figsize=(10, 10))

    #plt.subplot(4, 1, 1)
    #plt.plot(preprocess_speed_list, label='Preprocesamiento')
    #plt.xlabel('Tiempo')
    #plt.ylabel('Tiempo (ms)')
    #plt.title('Velocidad de preprocesamiento')

    plt.subplot(4, 1, 1)
    plt.plot(inference_speed_list, label='Inferencia')
    plt.xlabel('Tiempo')
    plt.ylabel('Tiempo (ms)')
    plt.title('Velocidad de inferencia')

    #plt.subplot(4, 1, 3)
    #plt.plot(postprocess_speed_list, label='Postprocesamiento')
    #plt.xlabel('Tiempo')
    #plt.ylabel('Tiempo (ms)')
    #plt.title('Velocidad de postprocesamiento')

    plt.subplot(4, 1, 2)
    plt.plot(fps_list, label='FPS')
    plt.xlabel('Tiempo')
    plt.ylabel('FPS')
    plt.title('FPS de la cámara')

    #plt.subplot(4, 1, 3)
    #plt.plot(people_count_list, label='Personas Detectadas', color='green')
    #plt.xlabel('Tiempo')
    #plt.ylabel('Número de personas')
    #plt.title('Cantidad de personas detectadas')

    plt.tight_layout()
    plt.show()