import math
import roslibpy
import numpy as np
import uuid

class control:

    def __init__(self, intr, ros):
        self.ros = ros
        self.intr = intr
        self.global_coordinates = []
        self.suscriber = roslibpy.Topic(self.ros, '/robot_pose', 'geometry_msgs/Pose')
        self.R = 0.05
        self.H = 1
        self.Q = 0.05
        self.P_x = 0
        self.x_filtered = 0
        self.K_x = 0
        self.P_y = 0
        self.y_filtered = 0
        self.K_y = 0
        self.prev_X = None
        self.prev_Y = None
        self.map_type = 'base_link'
        self.x = None
        self.y = None
        self.depth_frame = None
        self.theta = 0
        self.selected_id = 1
        self.relative = True
        self.mode = True
    
    def set_theta(self, theta):
        self.theta = theta

    def set_selected_id(self, id):
        self.selected_id = id
    
    def set_relative(self, boolean):
        self.relative = boolean
        print(f'Relative mode changed: {self.relative}')
    
    def set_mode(self, mode):
        self.mode = mode
        print(f'Mode changed: {self.mode}')

    

    def get_object_3d_coordinates(self):
    
        print('get_object_3d_coordinates')
        
        center_x = self.x
        center_y = self.y
        
        depth = self.depth_frame.get_distance(int(center_x), int(center_y))*1000

        Xtemp = depth*(center_x -self.intr.ppx)/self.intr.fx
        Ytemp = depth*(center_y -self.intr.ppy)/self.intr.fy
        Ztemp = depth                
        Xtarget = Xtemp - 35 
        Ytarget = -(Ztemp*math.sin(self.theta) + Ytemp*math.cos(self.theta))
        Ztarget = Ztemp*math.cos(self.theta) + Ytemp*math.sin(self.theta)                
        
        angle = -math.atan2(Xtarget, Ztarget) * 180/math.pi

        return Xtarget, Ytarget, Ztarget, angle
    
    def get_global_coordinate(self):
        self.global_coordinates = []
        self.suscriber.subscribe(self.get_message)
        while len(self.global_coordinates) == 0:pass
        self.suscriber.unsubscribe()
        self.global_coordinates = self.transform_global_coordinates()

    def get_message(self, message):
        if 'position' in message and 'orientation' in message:
            self.global_coordinates.append([message['position'], message['orientation']])
            #print(global_coordinates)


    def transform_global_coordinates(self):
        result = {
            'position': self.global_coordinates[-1][0],
            'orientation': self.global_coordinates[1][0]
        }
        return result
    
    def get_quadrant_from_quaternion(self):
        quaternion = self.global_coordinates['orientation']
        # Extraer los valores del cuaternión
        x = quaternion['x']
        y = quaternion['y'] 
        z = quaternion['z']
        w = quaternion['w']
        
        # Calcular el ángulo en radianes
        angle_rad = 2 * math.atan2(math.sqrt(x**2 + y**2 + z**2), w)
        
        # Convertir el ángulo a grados
        angle_deg = math.degrees(angle_rad)
        print('Grados: ' + str(angle_deg))
        # Calcular el seno y coseno del ángulo
        sin_angle = math.sin(angle_rad)
        cos_angle = math.cos(angle_rad)
        
        # Determinar el cuadrante basado en los signos de seno y coseno
        if sin_angle >= 0 and cos_angle >= 0:
            
            quadrant = 1
        elif sin_angle >= 0 and cos_angle < 0:
            quadrant = 2
        elif sin_angle < 0 and cos_angle < 0:
            quadrant = 3
        else:
            quadrant = 4
        
        return quadrant, angle_rad
    
    def apply_kalman_filter(self, x, y):

        self.K_x = self.P_x*self.H/(self.H*self.P_x*self.H+self.R)
        self.K_y = self.P_y*self.H/(self.H*self.P_y*self.H+self.R)
        self.x_filtered += self.K_x * (x - self.H * self.x_filtered)
        self.y_filtered += self.K_y * (y - self.H * self.y_filtered)
        self.P_x = (1-self.K_x*self.H)*self.P_x+self.Q
        self.P_y = (1-self.K_y*self.H)*self.P_y+self.Q

        return self.x_filtered, self.y_filtered

    def transform_coords(self, X, Y, Z, quadrant, angle_rad):
        if not self.relative:
            print('GLobal')
            #robot_x = self.global_coordinates['position']['x']
            #robot_y = self.global_coordinates['position']['y']
            #robot_z = self.global_coordinates['position']['z']
            
            #yaw = angle_rad
            
            # Apply rotation
            #X = X * math.cos(yaw) - Y * math.sin(yaw)
            #Y = X * math.sin(yaw) + Y * math.cos(yaw)
            #Z = Z
            
            # Apply translation
            #X += robot_x
            #Y += robot_y
            #Z += robot_z

            #self.map_type = 'map'

        else: self.map_type = 'base_link'
    
        

        if(quadrant == 1):
            X -= 0.4
            Y -= 0.4
        elif(quadrant == 2):
            X += 0.4
            Y -= 0.4
        elif(quadrant == 3):
            X += 0.4
            Y += 0.4
        elif(quadrant == 4):
            X -= 0.4
            Y += 0.4

        return X, Y, Z
    
    def get_cuaternion(self, angle):
        angle = math.radians(angle)
        half_angle = angle/2
        z = np.sin(half_angle)
        w = np.cos(half_angle)
        return z, w
    
    def control(self, depth_frame, track_id, x, y):
        self.x = x
        self.y = y
        self.depth_frame = depth_frame
        X, Y, Z, angle = self.get_object_3d_coordinates()
        y_global = -X / 1000
        z_global = Y / 1000
        x_global = Z / 1000

        print(f'X: {x_global}')
        print(f'Y: {y_global}')
        print(f'Z: {z_global}')

        #get_global_coordinate()
        #quadrant, angle_rad = get_quadrant_from_quaternion()
        if x_global is not None and x_global > 0.4 and x_global < 8 and int(self.selected_id) == track_id:
            
            x_global, y_global = self.apply_kalman_filter(x_global, y_global) #A ver que tal funciona aqui
            
            x_global, y_global, z_global = self.transform_coords(x_global, y_global, z_global, 1, angle_rad = 0)
            

            #if X < 0: X += 0.4
            #else: X -= 0.4

            
            print('Enviando (X, Y) filtradas:', x_global, y_global)
            print("Angle: " + str(angle))
            if x_global > 4: 
                vl = 1.2
                vr = 1
            else: 
                vl = 0.5
                vr = 0.5
            
            if self.mode == True:
                if -10 < angle < 10:
                    self.rosbridge(x_global, 0, vl, vr)
                    print("Lineal Sin Y")
                
                else:
                    
                    z, w = self.get_cuaternion(angle)
                    self.rosbridge(0, 0, 1, 1, z, w)
                    print("Lineal ROtando")
            else:
                print("Mode XY")
                self.rosbridge(x_global, y_global, vl, vr)

    def rosbridge(self, x, y, vl, vr, zx = 0.0, wy = 1.0):
        goal_id = str(uuid.uuid4())
        
        pose_stamped_msg = {
            'header': {
                'stamp': {
                    'secs': 0,
                    'nsecs': 0
                },
                'frame_id': self.map_type
            },
            'pose': {
                'position': {
                    'x': x,
                    'y': y,
                    'z': 0.0
                },
                'orientation': {
                    'x': 0,
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
                'collision_detection': False,
                'target_offset': 0.5,
                'respect_forbidden': True,
                'max_linear_speed': vl,
                'max_rotational_speed' : vr
            },


        }

        publisher = roslibpy.Topic(self.ros, '/move_base/goal', 'mir_nav_interface/MirMoveBaseActionGoal')
        publisher.publish(roslibpy.Message(move_base_goal_msg))
        