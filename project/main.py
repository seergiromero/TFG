import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from tracker import Tracker
import math
import cv2
from flask import Flask, render_template
from flask_socketio import SocketIO
from flask_cors import CORS
import base64
import roslibpy
import uuid
from shapely.geometry import LineString
from project.classreid import reidentification

# Configurar Flask y Socket
app = Flask(__name__, static_url_path='/static')
socketio = SocketIO(app)  
CORS(app, resources={r"/*": {"origins": "http://172.20.10.4:5000"}})

# Variables control botones aplicación web
start = False
detect = False
mode = True
focus = False
relative = True
YOLO_score = 0.8
count = 15
selected_id = 1

# Configurar la cámara RealSense D435
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
config.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 60)
profile = pipeline.start(config)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
align = rs.stream.color
align = rs.align(align)

# DeepSORT
tracker = Tracker()

# Configuración ROSBRIDGE
ros = roslibpy.Ros(host='192.168.12.20', port=9090)
suscriber = roslibpy.Topic(ros, '/robot_pose', 'geometry_msgs/Pose')
ros.run()

# Variables control
angle_rot = 15
coordinates = []
global_coordinates = []
map_type = 'base_link'
prev_ids = []

# Reidentificacion
reid = reidentification()

# Cargar el modelo YOLOv8
model = YOLO("yolov8n-pose.pt")


def rosbridge(xpos, ypos, zx = 0.0, wy = 1.0):
    global global_coordinates
    goal_id = str(uuid.uuid4())

    if zx == 0.0 and wy == 1.0 and not relative:
        zx = float(global_coordinates['orientation']['z'])
        wy = float(global_coordinates['orientation']['w'])
    
    pose_stamped_msg = {
        'header': {
            'stamp': {
                'secs': 0,
                'nsecs': 0
            },
            'frame_id': map_type
        },
        'pose': {
            'position': {
                'x': xpos,
                'y': ypos,
                'z': 0.0
            },
            'orientation': {
                'x': 0.0,
                'y': 0.0,
                'z': zx,
                'w': wy
            }
        }
    }

    move_base_goal_msg = {

        'goal_id': {
            'stamp' : {
                'secs': 0,
                'nsecs': 0
            },
            'id' : goal_id
            
        },

        'goal': {
            'move_task' : 1,
            'target_pose': {
                'header': pose_stamped_msg['header'],
                'pose': pose_stamped_msg['pose']
            },
            'clear_costmaps' : True,
            'timeout' : 60.0,
            'collision_detection': True,
            'collision_avoidance': False,
            'only_collision_detection': False,
            'target_offset': 0.5,
            'respect_forbidden': True,
            'max_linear_speed': 0.0,
            'max_rotational_speed' : 0.0
        },


    }

    publisher = roslibpy.Topic(ros, '/move_base/goal', 'mir_nav_interface/MirMoveBaseActionGoal')
    publisher.publish(roslibpy.Message(move_base_goal_msg))
    

@socketio.on('connect')
def handle_connect():
    print('Usuario connectado')

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

@socketio.on('start_video')
def start_video(state):
    global start
    start = state
    print(f'Sistema {"activo" if state else "inactivo"}')

@socketio.on('lineal_mode')
def lineal_mode(state):
    global mode
    mode = state
    print(f'Lineal mode esta {"activo" if state else "inactivo"}')

@socketio.on('Relative_tracking')
def Global_tracking(state):
    global relative
    relative = state
    print(f'Relative mode esta {"activo" if relative else "inactivo"}')

@socketio.on('focus_mode')
def focus_mode(state):
    global focus
    focus = state
    print(f'Focus mode esta {"activo" if focus else "inactivo"}')

@socketio.on('detect')
def start_detect(state):
    global detect
    detect = state
    print(f'Detect esta {"activo" if detect else "inactivo"}')

@app.route('/')
def index():
    return render_template('index.html')

def get_object_3d_coordinates(theta, depth_frame, x = None, y=None):
        
    center_x = x
    center_y = y
    
    depth = depth_frame.get_distance(int(center_x), int(center_y))*1000

    Xtemp = depth*(center_x -intr.ppx)/intr.fx
    Ytemp = depth*(center_y -intr.ppy)/intr.fy
    Ztemp = depth                
    Xtarget = Xtemp - 35 
    Ytarget = -(Ztemp*math.sin(theta) + Ytemp*math.cos(theta))
    Ztarget = Ztemp*math.cos(theta) + Ytemp*math.sin(theta)                
    
    angle = -math.atan2(Xtarget, Ztarget) * 180/math.pi

    return Xtarget, Ytarget, Ztarget, angle

def get_cuaternion(angle):
    angle = math.radians(angle)
    half_angle = angle/2
    z = np.sin(half_angle)
    w = np.cos(half_angle)
    return z, w


def get_quadrant_from_quaternion():
    quaternion = global_coordinates['orientation']

    x = quaternion['x']
    y = quaternion['y'] 
    z = quaternion['z']
    w = quaternion['w']
    
    angle_rad = 2 * math.atan2(math.sqrt(x**2 + y**2 + z**2), w)
    
    #angle_deg = math.degrees(angle_rad)

    sin_angle = math.sin(angle_rad)
    cos_angle = math.cos(angle_rad)
    
    if sin_angle >= 0 and cos_angle >= 0:        
        quadrant = 1
    elif sin_angle >= 0 and cos_angle < 0:
        quadrant = 2
    elif sin_angle < 0 and cos_angle < 0:
        quadrant = 3
    else:
        quadrant = 4
    
    return quadrant, angle_rad


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
    
    
def get_message(message):
    if 'position' in message and 'orientation' in message:
        global_coordinates.append([message['position'], message['orientation']])
        

def get_global_coordinate():
    global global_coordinates
    global_coordinates = []
    suscriber.subscribe(get_message)
    while len(global_coordinates) == 0: pass
    suscriber.unsubscribe()
    global_coordinates = transform_global_coordinates(global_coordinates[-1])

def transform_global_coordinates(coordinates):
    result = {
        'position': coordinates[0],
        'orientation': coordinates[1]
    }
    return result

def transform_coords(X, Y, Z, quadrant, angle_rad):
    global relative, map_type, angle_rot

    if mode:
        if(quadrant == 1):
            X -= 0.7
            
        elif(quadrant == 2):
            X += 0.7 
            
        elif(quadrant == 3):
            X += 0.7
            
        elif(quadrant == 4):
            X -= 0.7
            
    elif abs(Y) < 0.7:
        if(quadrant == 1):
            X -= 0.7
            
        elif(quadrant == 2):
            X += 0.7
            
        elif(quadrant == 3):
            X += 0.7
            
        elif(quadrant == 4):
            X -= 0.7
                       
    else:
        if(quadrant == 1):
            X -= 0.7
            Y -= 0.7
        elif(quadrant == 2):
            X += 0.7
            Y -= 0.7
        elif(quadrant == 3):
            X += 0.7
            Y += 0.7
        elif(quadrant == 4):
            X -= 0.7
            Y += 0.7

    if not relative:
        robot_x = global_coordinates['position']['x']
        robot_y = global_coordinates['position']['y']
        robot_z = global_coordinates['position']['z']
        
        yaw = angle_rad
               
        X = X * math.cos(yaw) - Y * math.sin(yaw)
        Y = X * math.sin(yaw) + Y * math.cos(yaw)
        Z = Z
        
        X += robot_x
        Y += robot_y
        Z += robot_z

        map_type = 'map'

    else: 

        map_type = 'base_link'
   
    return X, Y, Z

def control(depth_frame, track_id, x, y):

    X, Y, Z, angle = get_object_3d_coordinates(0, depth_frame, x, y)
    y_global = -X / 1000
    z_global = Y / 1000
    x_global = Z / 1000

    get_global_coordinate()
    quadrant, angle_rad = get_quadrant_from_quaternion()

    if x_global is not None and abs(x_global) > 0.7 and abs(x_global) < 10 and int(selected_id) == track_id:
        
        x_global, y_global, z_global = transform_coords(x_global, y_global, z_global, quadrant, angle_rad)
        
 
        if mode:
            
            if -angle_rot < angle < angle_rot:
                rosbridge(x_global, 0)
                   
            else:
                
                z, w = get_cuaternion(angle)
                rosbridge(0.0, 0.0, z, w)                
 
        else:
            rosbridge(x_global, y_global)

def detection():
    try:
        global start, count, selected_id, detect
        x = None
        y = None
        max_track_id = 0

        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

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
                                    control(depth_frame, track_id, x, y)
   
                        elif not focus:
                            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (155, 155, 155), 3)
                            cv2.putText(color_image, "Id" + str(track_id), (int(x1), int(y1) + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (155, 155, 155), 3)
            
            if not start:
                rosbridge(0,0,0,0)

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
    video_thread = threading.Thread(target=detection)
    video_thread.start()
    socketio.run(app, host='0.0.0.0', port=5000)