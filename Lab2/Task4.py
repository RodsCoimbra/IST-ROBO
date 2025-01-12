import curses
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
# Constants
DELTA_T = 0.02 # Time interval in seconds
TRUST_LAST_LTA_MEASUREMENT_TIME = 1 # Time in seconds to trust the last LTA measurement
ANGLE_INCREMENT = np.radians(2)  # Angle increment in radians
MAX_ANGLE = np.radians(30)  # Maximum steering angle in radians
WHEELBASE = 2.36  # Wheelbase in meters
FRONT_WIDTH = 1.35 # Front wheel width in meters
LANE_WIDTH = 4  # Lane width in meters
MIN_VEL = 14
MAX_VEL = 32
NUM_STEPS_LOOK_AHEAD = 4
DISTANCE_THRESHOLD_LTA = 1.5
WINDOW_SIZE_MEAN = 5
NOISE_STD = 0.1
SIM_TIME = 7
# Car wheels positions (Front Right, Front Left, Back Left, Back Right)
CAR_CORNERS = np.array([(WHEELBASE, -FRONT_WIDTH/2),
                        (WHEELBASE, FRONT_WIDTH/2),
                        (0, FRONT_WIDTH/2),
                        (0, -FRONT_WIDTH/2)]) 

SENSOR_POSITIONS = np.array([WHEELBASE, 0])
np.random.seed(42)
class Car:
    def __init__(self, velocity):
        # CAR Parameters
        self.car_position = (LANE_WIDTH/2 + 1.2, 0)  
        self.theta = np.radians(91)
        self.phi = 0
        self.velocity = velocity
        
        # Simulation Parameters
        self.num_seconds_sim = 0

        # DISTANCE TO LANE
        self.distance_left = []
        self.distance_right =[]
        self.distance_left_mean = deque([],maxlen=WINDOW_SIZE_MEAN)
        self.distance_right_mean = deque([],maxlen=WINDOW_SIZE_MEAN)
        self.distance_left_gt = []
        self.distance_right_gt = []
        #MAP LANE LIMITS
        self.y_trajectory = np.linspace(-2,200,1000)
        self.x_left_trajectory = np.zeros(1000)
        self.x_right_trajectory = self.x_left_trajectory + LANE_WIDTH
    
        #Subplots    
        self.theta_list = [self.theta]
        self.position_plot = [self.car_position]
        self.time = []
    
    def start_simulator(self, stdscr):
        self.stdscr = stdscr
        curses.cbreak()
        self.stdscr.nodelay(True)  # Make getch non-blocking
        while self.num_seconds_sim < SIM_TIME:
            self.num_seconds_sim += DELTA_T
            self.car_position = self.next_car_position()
            self.position_plot.append(self.car_position)
            self.theta_list.append(self.theta)
            self.sensor_position = rotation(SENSOR_POSITIONS, self.theta) + self.car_position
            self.time.append(self.num_seconds_sim)
            self.distance_to_lane()
            
        self.display_final()
             
    def next_car_position(self):
        dx, dy = self.get_direction()
        delta_x = dx * DELTA_T
        delta_y = dy * DELTA_T
        self.theta += self.velocity * np.tan(self.phi)/WHEELBASE * DELTA_T  
        x = self.car_position[0] + delta_x
        y = self.car_position[1] + delta_y
        return (x,y)
    
    def get_direction(self):
        dx = self.velocity * np.cos(self.theta)
        dy = self.velocity * np.sin(self.theta)
        return dx, dy
    
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

        # To improve performance, we only consider the segment of the trajectory that is in front of the sensor and to a maximum of 10 meters ahead
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

        left_lane_point, left_lane_dist = self.find_intersection(right_laser_angle, self.x_left_trajectory, self.y_trajectory)
        if left_lane_point is None:     
            left_lane_dist = -1
        self.distance_left_mean.append(float(left_lane_dist) + np.random.normal(0, NOISE_STD))
        self.distance_left.append(np.mean(self.distance_left_mean))
        self.distance_left_gt.append(float(left_lane_dist))
        
        right_lane_point, right_lane_dist = self.find_intersection(left_laser_angle, self.x_right_trajectory, self.y_trajectory)
        
        if right_lane_point is None:
            right_lane_dist = -1
        self.distance_right_mean.append(float(right_lane_dist) + np.random.normal(0, NOISE_STD))
        self.distance_right.append(np.mean(self.distance_right_mean))
        self.distance_right_gt.append(float(right_lane_dist))
        
        
    def display_final(self):      
        plt.figure(figsize=(8, 6))
        self.create_corners_plot()
        position_plot = np.array(self.position_plot)
        plt.plot(self.x_left_trajectory, self.y_trajectory, 'k--', label='Lane Limit')
        plt.plot(self.x_right_trajectory, self.y_trajectory, 'k--')
        plt.legend(loc='upper left')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.xlim(LANE_WIDTH/2 - 5, LANE_WIDTH/2 + 5)
        plt.ylim(position_plot[:,1].min() - 2, position_plot[:,1].max() + 5)
        plt.tight_layout()

        plt.figure(figsize=(8, 6))
        plt.plot(self.time, self.distance_right, label='Right Distance', color='#7CA5B8', alpha=0.8)
        plt.plot(self.time, self.distance_left, label='Left Distance', color='#E03616', alpha=0.8)
        plt.plot(self.time, self.distance_right_gt, label='Right Distance Ground Truth', linestyle='--', color='#7CA5B8')
        plt.plot(self.time, self.distance_left_gt, label='Left Distance Ground Truth', linestyle='--', color='#E03616')
        plt.legend(loc='upper left')
        plt.xlabel('Time (s)')
        plt.ylabel('Distance (m)')
        plt.tight_layout()

        plt.show()
       
    def create_corners_plot(self):
        for i in range(1, len(self.position_plot), 10):
            corners = self.corners_car(self.position_plot[i], self.theta_list[i])
            chassis_car = np.append(corners, [corners[0]], axis=0)
            plt.plot(chassis_car[:,0],chassis_car[:,1], color='blue')
            
        corners = self.corners_car(self.position_plot[i], self.theta_list[i])
        chassis_car = np.append(corners, [corners[0]], axis=0)
        plt.plot(chassis_car[:,0],chassis_car[:,1], color='blue', label="Car's Chassis") 
       
def rotation(point, theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ point
    
    
if __name__ == "__main__":
    car = Car(20)        
    curses.wrapper(car.start_simulator)


