import curses
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque 

# Constants
DELTA_T = 0.02 # Time interval in seconds
DISPLAY_DELTA = 5 * DELTA_T # Display update interval in seconds
TRUST_LAST_LTA_MEASUREMENT_TIME = 1 # Time in seconds to trust the last LTA measurement
ANGLE_INCREMENT = np.radians(1)  # Angle increment in radians
MAX_ANGLE = np.radians(30)  # Maximum steering angle in radians
WHEELBASE = 2.36  # Wheelbase in meters
FRONT_WIDTH = 1.35 # Front wheel width in meters
WHEEL_RADIUS = 0.330  # Wheel radius in meters
LANE_WIDTH = 4  # Lane width in meters
MIN_VEL = 10
MAX_VEL = 30
FUTURE_LOOK_AHEAD = 0.5 # Future look ahead in seconds
NUM_STEPS_LOOK_AHEAD = 4
DISTANCE_THRESHOLD_LTA = 1.5
WINDOW_SIZE_MEAN = 1
LENGTH_CURVE = 15.7
NOISE_STD = 0.00

# Car wheels positions (Front Right, Front Left, Back Left, Back Right)
CAR_CORNERS = np.array([(WHEELBASE, -FRONT_WIDTH/2),
                        (WHEELBASE, FRONT_WIDTH/2),
                        (0, FRONT_WIDTH/2),
                        (0, -FRONT_WIDTH/2)]) 

SENSOR_POSITIONS = np.array([WHEELBASE, 0])

class Car:
    def __init__(self, velocity):
        self.car_position = (LANE_WIDTH/2, -10)  
        self.distance_left = deque([self.car_position[0] - FRONT_WIDTH/2],maxlen=100)
        self.distance_right = deque([LANE_WIDTH - (self.car_position[0] + FRONT_WIDTH/2)],maxlen=100)
        self.theta = np.radians(90)
        self.phi = 0
        self.velocity = velocity
        self.step_size_look_ahead = int(FUTURE_LOOK_AHEAD/(DELTA_T*NUM_STEPS_LOOK_AHEAD))
        self.controller = controller()
        self.use_lta = 0
        #self.last_measurments = deque([0] * WINDOW_SIZE_MEAN,maxlen=WINDOW_SIZE_MEAN)
        self.num_seconds_sim = 0
        self.dist_last_iteration = np.array([0,0])
        y_curve = np.linspace(0,LENGTH_CURVE, 1000) 
        self.y_trajectory = np.concatenate((np.linspace(-25,0,250), y_curve, np.linspace(LENGTH_CURVE, LENGTH_CURVE+25, 250)))
        self.x_left_trajectory = np.concatenate((np.zeros(250), 2 * np.sin(y_curve/2.5), np.zeros(250)))
        self.x_right_trajectory = self.x_left_trajectory + LANE_WIDTH
        self.left_lane_point = None
        self.right_lane_point = None
        
        plt.ion() 
        _, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 8))
        #_, (self.ax3, self.ax4) = plt.subplots(1, 2, figsize=(15, 8))
        
        self.last_update_time = 0
        self.last_update_time_sim = 0 
        self.plot_phi = deque([self.phi],maxlen=100)
        self.plot_theta = deque([self.theta],maxlen=100)
        self.warning_img = plt.imread('warning.png') 
        self.lta_working_img = plt.imread('lta_working.webp')
        self.unavailable = plt.imread('unavailable.png')
    
    def start_simulator(self, stdscr):
        self.stdscr = stdscr
        curses.cbreak()
        self.stdscr.nodelay(True)  # Make getch non-blocking
        while True:
            if(self.joystick()):
                break
            
            # Periodic update every Delta_T seconds
            current_time = time.time()
            
            if current_time - self.last_update_time >= DELTA_T:
                self.car_position = self.next_car_position()
                self.last_update_time = current_time
                self.corners = self.corners_car(self.car_position, self.theta)
                self.sensor_position = rotation(SENSOR_POSITIONS, self.theta) + self.car_position
                self.distance_to_lane()
                self.danger_zone()  
                #Uses the LTA controller
                if self.use_lta > 0:
                    #ERROR  Orientation from Car position to middle lane point- Current orientation
                    predicted_error = np.arctan2(self.middle_lane_point[1] - self.car_position[1], self.middle_lane_point[0] - self.car_position[0]) - self.theta
                    #self.last_measurments.append(predicted_error)
                    #predicted_error = np.mean(self.last_measurments)
                    self.phi = self.controller.pid(predicted_error)
                     
                    
                #Simulation update every DISPLAY_DELTA seconds
                if current_time - self.last_update_time_sim >= DISPLAY_DELTA:
                    self.num_seconds_sim += DISPLAY_DELTA 
                    self.plot_theta.append(self.theta)
                    self.plot_phi.append(self.phi)
                    self.last_update_time_sim = current_time
                    self.display_simulation()
                        
    def joystick(self):
        key = self.stdscr.getch()  
        if key == curses.KEY_LEFT:
            self.change_angle(ANGLE_INCREMENT)
        elif key == curses.KEY_RIGHT:
            self.change_angle(-ANGLE_INCREMENT)
        elif key == curses.KEY_UP:
            self.change_velocity(3)
        elif key == curses.KEY_DOWN:
            self.change_velocity(-3)
        elif key == ord('x'):
            self.stdscr.addstr("Exiting...\n")
            plt.ioff()
            plt.close('all')
            return 1
        
        return 0
      
    def change_velocity(self, increment):
        self.velocity += increment
        self.velocity = np.clip(self.velocity, MIN_VEL, MAX_VEL)
    
    def change_angle(self, angle):
        self.phi += angle
        self.phi = np.clip(self.phi, -MAX_ANGLE, MAX_ANGLE)
        
    def next_car_position(self):
        dx, dy = self.get_direction()
        delta_x = dx * DELTA_T
        delta_y = dy * DELTA_T
        self.theta += self.velocity * np.tan(self.phi)/WHEELBASE * DELTA_T  
        x = self.car_position[0] + delta_x
        y = self.limit_inside_display(self.car_position[1] + delta_y)
        return (x,y)
     
    def limit_inside_display(self,value):
        min_value = -15
        max_value = LENGTH_CURVE + 15
        if value > max_value:
            value = min_value + (value - max_value)
        elif value < min_value:
            value = max_value - (min_value - value)
        
        return value
    
    def get_direction(self):
        dx = self.velocity * np.cos(self.theta)
        dy = self.velocity * np.sin(self.theta)
        return dx, dy
    
    def danger_zone(self):
        if self.use_lta == -1:
            return
        
        if (self.distance_right[-1] < DISTANCE_THRESHOLD_LTA or self.distance_left[-1] < DISTANCE_THRESHOLD_LTA):
            self.use_lta = 2
            return
        
        theta = self.theta 
        car_position = [*self.car_position]
        for _ in range(0, NUM_STEPS_LOOK_AHEAD):
            car_position[0] += self.velocity * np.cos(theta) * DELTA_T * self.step_size_look_ahead
            car_position[1] = self.limit_inside_display(car_position[1] + self.velocity * np.sin(theta) * DELTA_T * self.step_size_look_ahead)
            theta += self.velocity * np.tan(self.phi)/WHEELBASE * DELTA_T * self.step_size_look_ahead       
            corners = self.corners_car(car_position, theta, just_front = True)
            
            start_idx_right = np.searchsorted(self.y_trajectory, corners[0][1], side="left")
            start_idx_right = np.argmin(np.abs(self.y_trajectory[max(0,start_idx_right-1):min(len(self.y_trajectory),start_idx_right+1)] - corners[0][1]))
            start_idx_left = np.argmin(np.abs(self.y_trajectory[max(0,start_idx_right-10):min(len(self.y_trajectory),start_idx_right+10)] - corners[1][1])) # -10 and +10 to be computationally faster, as both should have similar values
            if (corners[0][0] > self.x_right_trajectory[start_idx_right] or corners[1][0] < self.x_left_trajectory[start_idx_left]):
                self.use_lta = 1
                return
        self.use_lta = 0
        return
    
    def corners_car(self, car_position, theta, just_front = False):
        corners = []
        num_corners = 2 if just_front else 4
        for i in range(num_corners):
            x, y = rotation(CAR_CORNERS[i], theta) + car_position
            corners.append([x,y])
            
        return np.array(corners)
        

    def find_intersection(self, angle, x_trajectory, y_trajectory):
        sensor_x, sensor_y = self.sensor_position
        x_direction, y_direction = np.cos(angle), np.sin(angle)
        closest_dist = float('inf')
        closest_point = None

        # Filter segments based on ray direction
        start_idx = np.searchsorted(y_trajectory, sensor_y, side="left") - 1
        end_idx = np.searchsorted(y_trajectory, sensor_y + 10, side="right")
        segments = zip(x_trajectory[start_idx:end_idx], x_trajectory[start_idx+1:end_idx+1], y_trajectory[start_idx:end_idx], y_trajectory[start_idx+1:end_idx+1])

        for x1, x2, y1, y2 in segments:
            sx, sy = x2 - x1, y2 - y1
            denom = x_direction * sy - y_direction * sx
            if abs(denom) < 1e-9:  # Parallel lines
                continue

            # Calculate intersection parameters
            t = (sx * (sensor_y - y1) - sy * (sensor_x - x1)) / denom
            u = (x_direction * (sensor_y - y1) - y_direction * (sensor_x - x1)) / denom

            # Check if the intersection is valid
            if t >= 0 and 0 <= u <= 1:
                ix, iy = sensor_x + t * x_direction, sensor_y + t * y_direction
                dist = np.hypot(ix - sensor_x, iy - sensor_y)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_point = (ix, iy)

        return closest_point, closest_dist


    def distance_to_lane(self):
        right_laser_angle = self.theta + np.pi/5
        left_laser_angle  = self.theta - np.pi/5
        self.dist_last_iteration+=1

        left_lane_point, left_lane_dist = self.find_intersection(right_laser_angle, self.x_left_trajectory, self.y_trajectory)
        if left_lane_point is not None:     
            self.left_lane_point = left_lane_point
            self.distance_left.append(left_lane_dist)
            self.dist_last_iteration[0] = 0
        
        right_lane_point, right_lane_dist = self.find_intersection(left_laser_angle, self.x_right_trajectory, self.y_trajectory)
        
        if right_lane_point is not None:
            self.right_lane_point = right_lane_point
            self.distance_right.append(right_lane_dist)
            self.dist_last_iteration[1] = 0
        
        
        if np.any(self.dist_last_iteration > TRUST_LAST_LTA_MEASUREMENT_TIME/DELTA_T):
            self.use_lta = -1
            return
        elif self.use_lta == -1:
            self.use_lta = 0
        
        self.middle_lane_point = (self.left_lane_point[0] + self.right_lane_point[0])/2, (self.left_lane_point[1] + self.right_lane_point[1])/2
        
        
    def display_simulation(self):
        self.ax1.clear()
        self.ax2.clear()
        
        x,y = self.car_position
        chassis_car = np.append(self.corners, [self.corners[0]], axis=0)
        self.ax1.plot(chassis_car[:,0],chassis_car[:,1], 'b', label='Car')
        dx, dy = self.get_direction()
        self.ax1.arrow(x, y, dx*0.075, dy*0.075, head_width=0.2, head_length=0.1, fc='k', ec='k')
        
        #MAP LANE LIMITS
        self.ax1.plot(self.x_left_trajectory[0:-1:30], self.y_trajectory[0:-1:30], 'k--', label='Lane Limit')
        self.ax1.plot(self.x_right_trajectory[0:-1:30], self.y_trajectory[0:-1:30], 'k--')
        
        #Sensor and sensor detection
        self.ax1.scatter(self.sensor_position[0], self.sensor_position[1], color='r', label='Sensor')   
        if self.left_lane_point is not None and self.right_lane_point is not None and self.use_lta != -1:
            self.ax1.scatter(*self.left_lane_point, s=50, color='#7CA5B8')
            self.ax1.scatter(*self.right_lane_point, s=50, color='#7CA5B8')
            self.ax1.plot([self.sensor_position[0], self.right_lane_point[0]], [self.sensor_position[1], self.right_lane_point[1]], 'b--')
            self.ax1.plot([self.sensor_position[0], self.left_lane_point[0]], [self.sensor_position[1], self.left_lane_point[1]], 'b--')
            self.ax1.plot([self.left_lane_point[0], self.right_lane_point[0]], [self.left_lane_point[1], self.right_lane_point[1]], 'r--')
            self.ax1.scatter(*self.middle_lane_point, s=50, color='r', label='Reference Point')
        
        self.img_display_lta(x, y)
             
        
        data = f"Velocity: {self.velocity}\nSteering Angle: {round(np.degrees(self.phi),3)}\n Time: {round(self.num_seconds_sim,2)}"
        self.ax1.text(
            0.05, 0.05,        
            data,
            transform=self.ax1.transAxes,
            horizontalalignment='left',
            verticalalignment='bottom'
        )
        
        self.ax1.set_xticklabels([])
        self.ax1.set_yticklabels([])
        self.ax1.set_xlim(x - 5, x + 5)
        self.ax1.set_ylim(y - 2.5, y + 7.5)
        self.ax1.legend(loc='upper left') 
        self.ax2.plot(np.arange(len(self.distance_left)), self.distance_left, label='Distance to left lane')
        self.ax2.plot(np.arange(len(self.distance_right)), self.distance_right, label='Distance to right lane')
        self.ax2.legend(loc='upper left')
        self.ax2.set_ylim(-4, 12)
        
        # self.ax3.clear()
        # self.ax4.clear()
        # self.ax3.plot(np.arange(len(self.plot_theta)), np.degrees(self.plot_theta), label='Theta', color='b')
        # self.ax4.plot(np.arange(len(self.plot_phi)), np.degrees(self.plot_phi), label='Phi', color='r')
        # self.ax3.set_ylim(-180, 180)
        # self.ax4.set_ylim(-30, 30)
        
        plt.draw()
        plt.pause(0.001)
        
    def img_display_lta(self, x, y):
        if self.use_lta == 0:
            return
        
        if self.use_lta == -1:
            self.ax1.imshow(self.unavailable, extent=[x + 3.5, x + 4.5, y + 6 , y + 7])
        
        if self.use_lta == 1:
            self.ax1.imshow(self.lta_working_img, extent=[x + 2.5, x + 5.5, y + 4.0 , y + 5.5])
           
        if self.use_lta == 2: 
            self.ax1.imshow(self.lta_working_img, extent=[x + 2.5, x + 5.5, y + 4.0 , y + 5.5])
            self.ax1.imshow(self.warning_img, extent=[x + 3.5, x + 4.5, y + 6 , y + 7])
              

class controller:
    def __init__(self):
        self.kp = 2 #0.5
        self.ki = 0.05#.01 #0.1
        self.kd = 0.1#.06 #0.05
        self.I = 0
        self.error = 0
        self.reference = 0
        self.previous_error = 0
        
    def pid(self, error):
        P = self.kp * error
        D = self.kd * (error - self.previous_error) 
        angle = P + self.I + D
        clip_angle = np.clip(angle, -MAX_ANGLE, MAX_ANGLE)
        self.I += self.ki * error 
        self.previous_error = error        
        return clip_angle
       
       
def rotation(point, theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ point
    
    
if __name__ == "__main__":
    car = Car(20)        
    curses.wrapper(car.start_simulator)


