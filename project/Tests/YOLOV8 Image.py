import cv2
import random
from ultralytics import YOLO


# Cargar el modelo YOLOv8
model = YOLO("yolov8n-pose.pt")

# Cargar la imagen deseada
image_path = "people.jpg"
color_image = cv2.imread(image_path)

# Hacer inferencia con YOLOv8
results = model(color_image)

# Colores aleatorios para las cajas de detección
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]

# Mostrar los resultados
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
        cv2.putText(color_image, "Id" + str(id), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX,4, (colors[id % len(colors)]),5)

# Mostrar la imagen con los objetos detectados
cv2.imshow('Imagen', color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
