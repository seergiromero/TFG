import cv2
import numpy as np
import random
import pyrealsense2 as rs
from ultralytics import YOLO
from tracker import Tracker
import math
import time
import cv2
from flask import Flask, render_template
from flask_socketio import SocketIO
from flask_cors import CORS
import matplotlib.pyplot as plt

import base64
import os
import roslibpy
import uuid
from decimal import Decimal, ROUND_HALF_UP
from shapely.geometry import LineString
from classreid import reidentification

app = Flask(__name__, static_url_path='/static')
socketio = SocketIO(app) 
CORS(app, resources={r"/*": {"origins": "http://192.168.1.14:5000"}})

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
config.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 60)
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
profile = pipeline.start(config)
tracker = Tracker()
fps_list = []
process_time_list = []
selected_id = 1
YOLO_score = 0.85
start = False
detect = False
mode = True
focus = False
relative = True
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
reid = reidentification()
map_type = 'base_link'
count = 15
max_track_id = 0
align = rs.stream.color
align = rs.align(align)
prev_ids = []
model = YOLO("yolov8n-pose.pt")


@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('send_number')
def handle_number(number):
    global selected_id
    selected_id = number
    print("Número recibido:", number)

@socketio.on('get_percentage')
def get_percentage(percentage):
    global YOLO_score
    YOLO_score = percentage
    print("Score recibido:", percentage)

@socketio.on('get_reid')
def get_reid(reid):
    global count
    count = reid
    print("Count recibido:", reid)


@socketio.on('start_stream')
def start_stream(state):
    global start, fps_list, process_time_list
    start = state
    
    print(f'System {"active" if state else "inactive"}')

@socketio.on('lineal_mode')
def lineal_mode(state):
    global mode
    mode = state

@socketio.on('Relative_tracking')
def Global_tracking(state):
    global relative
    relative = state

@socketio.on('focus_mode')
def focus_mode(state):
    global focus
    focus = state
    if state == True:
        plt.figure(figsize=(12, 6))
        
        times = [i / 60 for i in range(len(fps_list))]  

        mean_fps = np.mean(fps_list)
        mean_process_time = np.mean(process_time_list)

   
        print(f'Media de FPS: {mean_fps:.2f}')
        print(f'Media del tiempo de procesamiento: {mean_process_time:.4f} segundos')
        
        plt.subplot(2, 1, 1)
        plt.plot(times, fps_list, label='FPS')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('FPS')
        plt.title('FPS a lo largo del tiempo')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(times, process_time_list, label='Tiempo de procesamiento (s)', color='r')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Tiempo de procesamiento (s)')
        plt.title('Tiempo de procesamiento')
        plt.legend()

        plt.tight_layout()
        plt.show()

@socketio.on('detect')
def start_detect(state):
    global detect
    detect = state
    print(f'Detect is now {"active" if focus else "inactive"}')

@app.route('/')
def index():
    return render_template('index.html')


def get_pose(result, index):
    keypoints = result.keypoints 
    if keypoints is not None and len(keypoints) > 0:
        keypoints = keypoints[index].xy.cpu().numpy()  
        for kp in keypoints:
            if len(kp) >= 2:
                selected_values = [kp[6], kp[5], kp[12], kp[11]]
                line1 = LineString([selected_values[0], selected_values[3]])
                line2 = LineString([selected_values[1], selected_values[2]])
                intersection = line1.intersection(line2)
                if intersection.is_empty or intersection.x == 0.0 or intersection.y == 0.0:
                    return None, None
                else:
                    return float(intersection.x), float(intersection.y)
            return None, None
    

def video_stream():
    try:
        global start, count, selected_id, detect, fps_list, process_time_list
        x = None
        y = None
        max_track_id = 0

        while True:
            start_time = time.time()

            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()

            color_image = np.asanyarray(color_frame.get_data())
            results = model(color_image)

            prev_ids_copy = prev_ids.copy()
            prev_ids.clear()
            if detect:
                for result in results:
                    detections = []
                    pose_results = []
                    
                    for i, r in enumerate(result.boxes.data.tolist()):
                        x1, y1, x2, y2, score, _ = r
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        if score > float(YOLO_score):
                            detections.append([x1, y1, x2, y2, score])
                            pose_x, pose_y = get_pose(result, i)
                            pose_results.append((pose_x, pose_y))

                    tracker.update(color_image, detections)
                    for i, track in enumerate(tracker.tracks):
                        bbox = track.bbox
                        x1, y1, x2, y2 = bbox
                        track_id = track.track_id
                        roi = color_image[int(y1):int(y2), int(x1):int(x2)]
                        
                        if (track_id not in prev_ids_copy and max_track_id < track_id) or tracker.get_id(track_id) == 0  and not focus:
                            features = reid.extract_features_from_roi(roi)
                            if features is not None:
                                reid.reidentification_process(features, track_id, tracker, int(count))
                        
                        if max_track_id <= track_id:
                            max_track_id = track_id
                        prev_ids.append(track_id)
                        track_id = tracker.get_id(track_id)

                        if track_id is not None and int(track_id) == int(selected_id):
                            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                            cv2.putText(color_image, "Id" + str(track_id), (int(x1), int(y1) + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 3)
                            if i < len(pose_results):
                                x, y = pose_results[i]
                                if x is not None and y is not None and x1 < x < x2 and y1 < y < y2 and start:
                                    cv2.circle(color_image, (int(x), int(y)), 5, (0, 255, 0), -1)
                                    
                        elif not focus:
                            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (155, 155, 155), 3)
                            cv2.putText(color_image, "Id" + str(track_id), (int(x1), int(y1) + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (155, 155, 155), 3)
                end_time = time.time()
                process_time = end_time - start_time
                fps = 1 / process_time

                fps_list.append(fps)
                process_time_list.append(process_time)


            _, buffer = cv2.imencode('.jpg', color_image)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('video_frame', frame_base64)    

                       

            cv2.imshow('Imagen', color_image)
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        pipeline.stop()


if __name__ == '__main__':
    import threading
    video_thread = threading.Thread(target=video_stream)
    video_thread.start()
    socketio.run(app, host='0.0.0.0', port=5000)