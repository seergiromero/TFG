import pyrealsense2 as rs
import cv2
import numpy as np

# Configurar la cámara RealSense D435
pipeline = rs.pipeline()
config  = rs.config()
config.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 60)
config.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 60)
pipeline.start(config)
align = rs.stream.color
align = rs.align(align)

while True:
        
    # Capturar un frame de la cámara RealSense
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    # Convertir de frame a matriz
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Aplicar mapa de color a la imagen de profundidad
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,alpha=0.2), cv2.COLORMAP_JET)

    # Visualizar imagenes                                              
    cv2.imshow('RGB', color_image)
    cv2.imshow('Depth', depth_image)
    cv2.imshow('Color Depth', depth_colormap)

    if cv2.waitKey(1) == ord('q'):
        break

pipeline.stop()       