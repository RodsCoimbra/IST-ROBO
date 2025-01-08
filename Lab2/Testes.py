import curses
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque 

# Constants
DELTA_T = 0.05  # Time interval in seconds
ANGLE_INCREMENT = np.radians(9)  # Angle increment in radians
MAX_ANGLE = np.radians(30)  # Maximum steering angle in radians
WHEELBASE = 2.36  # Wheelbase in meters
FRONT_WIDTH = 1.35 # Front wheel width in meters
WHEEL_RADIUS = 0.330  # Wheel radius in meters
LANE_WIDTH = 3.5  # Lane width in meters
MIN_VEL = 10
MAX_VEL = 30
TIME_SIM = 30
NOISE_STD = 0.0


INITIAL_POSITION = -1.5
INITIAL_THETA = 95
INITIAL_PHI = 0
VELOCITY = 30
FUTURE_LOOK_AHEAD = 1 # Future look ahead in seconds
NUM_STEPS_LOOK_AHEAD = 4
THRESHOLD_LTA = 0.01
DISTANCE_THRESHOLD_LTA = 0.5
USE_ALWAYS_LTA = 1
WINDOW_SIZE_MEAN = 1
np.random.seed(42)

class Car:
    def __init__(self, wheelbase, front_width, wheel_radius):
        self.car_position = (LANE_WIDTH/2 + INITIAL_POSITION, 0)  
        self.plot_car = []
        self.plot_error = []
        self.distance_left = [self.car_position[0] - FRONT_WIDTH/2]
        self.distance_right = [LANE_WIDTH - (self.car_position[0] + FRONT_WIDTH/2)]
        self.wheel_radius = wheel_radius
        self.front_width = front_width
        self.theta = np.radians(INITIAL_THETA)
        self.phi = np.radians(INITIAL_PHI)
        self.plot_phi = []
        self.plot_theta = []
        self.plot_activation_lta = []
        self.velocity = VELOCITY
        self.wheelbase = wheelbase
        self.step_size_look_ahead = int(FUTURE_LOOK_AHEAD/(DELTA_T*NUM_STEPS_LOOK_AHEAD)) 
        self.controller = controller()
        self.use_lta = 0
        self.last_measurments = deque([0] * WINDOW_SIZE_MEAN,maxlen=WINDOW_SIZE_MEAN)
        self.error_lta = deque([],maxlen=100)
        _,(self.ax1, self.ax2) = plt.subplots(1,2)
        _, (self.ax3, self.ax4) = plt.subplots(1, 2)
    
    def start_simulator(self):
        i = 0
        while i < TIME_SIM/DELTA_T: #Run for 20 seconds
            i += 1
            self.car_position = self.next_car_position()
            self.plot_car.append(self.car_position)
            self.corners = self.corners_car(self.car_position)
            self.distance_to_lane()
            self.plot_activation_lta.append(self.use_lta)
            if  i % 20 and not USE_ALWAYS_LTA:
                self.danger_zone(LANE_WIDTH)
            if self.use_lta or USE_ALWAYS_LTA:
                predicted_error = self.distance_right[-1] - self.distance_left[-1] 
                self.last_measurments.append(predicted_error)
                desired_phi = self.controller.pid(np.mean(self.last_measurments))
                self.phi = desired_phi if abs(desired_phi - self.phi) <= ANGLE_INCREMENT else self.phi + np.sign(desired_phi - self.phi) * ANGLE_INCREMENT
                self.error_lta.append(predicted_error)
                if sum(abs(x) for x in self.error_lta) < THRESHOLD_LTA and len(self.error_lta) == self.error_lta.maxlen:
                    self.use_lta = 0
                    self.error_lta.clear()
                        
            self.plot_phi.append(self.phi)
            self.plot_error.append(np.median(self.last_measurments))
        self.display_simulation()
    
    def corners_car(self, center_position, just_front = False):
        corners = []
        # Order - Front Right, Front Left, Back Left, Back Right
        angles = [self.theta] if just_front else [self.theta, self.theta + np.pi]
        for theta in angles:
            x = center_position[0] +FRONT_WIDTH/2 * np.sin(theta) + WHEELBASE/2 * np.cos(theta)
            y = center_position[1] +WHEELBASE/2 * np.sin(theta) - FRONT_WIDTH/2 * np.cos(theta)
            corners.append(np.array([x, y]))
            x = center_position[0] -FRONT_WIDTH/2 * np.sin(theta) + WHEELBASE/2 * np.cos(theta)
            y = center_position[1] +WHEELBASE/2 * np.sin(theta) + FRONT_WIDTH/2 * np.cos(theta)
            corners.append(np.array([x, y]))
        
        return corners
    
    def distance_to_lane(self):
        """"Only the front corners are considered"""
        Front_right, Front_left = self.corners[0], self.corners[1]
        self.distance_left.append(Front_left[0] + np.random.normal(0, NOISE_STD))
        self.distance_right.append(LANE_WIDTH - Front_right[0] + np.random.normal(0, NOISE_STD))
        
    def next_car_position(self):
        dx, dy = self.get_direction()
        delta_x = dx * DELTA_T
        delta_y = dy * DELTA_T
        self.theta += self.velocity * np.sin(self.phi)/self.wheelbase * DELTA_T 
        self.plot_theta.append(self.get_theta()) 
        x = self.car_position[0] + delta_x
        y = self.car_position[1] + delta_y
        return (x,y)
    
    def get_theta(self):
        theta = np.rad2deg(self.theta)
        return theta % 360
    
    def get_direction(self):
        dx = self.velocity * np.cos(self.theta) * np.cos(self.phi)
        dy = self.velocity * np.sin(self.theta) * np.cos(self.phi)
        return dx, dy

    def display_simulation(self):
        x,y = zip(*self.plot_car)
        self.ax1.scatter(x,y, label='Car Position', s= 10)
        self.ax1.plot([0, 0], [0, self.car_position[1] + 10], 'k--', label='Lane Limit')
        self.ax1.plot([LANE_WIDTH, LANE_WIDTH], [0, self.car_position[1] + 10], 'k--')
        self.ax1.plot([LANE_WIDTH/2, LANE_WIDTH/2], [0, self.car_position[1] + 10], 'b--', label='Path')
        self.ax1.legend(loc='upper left') 
        self.ax2.plot(np.linspace(0,TIME_SIM,len(self.plot_error)), self.plot_error, label='Error')
        self.ax2.plot(np.linspace(0,TIME_SIM,len(self.distance_left)), self.distance_left, label='Distance Left')
        self.ax2.plot(np.linspace(0,TIME_SIM,len(self.distance_right)), self.distance_right, label='Distance Right')
        self.ax2.legend(loc='upper left')
        self.ax3.scatter(np.linspace(0,TIME_SIM,len(self.plot_phi)), self.plot_phi, label='Phi Value', s= 10)
        self.ax3.plot([0,TIME_SIM], [MAX_ANGLE, MAX_ANGLE], 'r--', label='Max Angle')
        self.ax3.plot([0,TIME_SIM], [-MAX_ANGLE, -MAX_ANGLE], 'k--', label='Min Angle')
        self.ax3.plot([0,TIME_SIM], [0, 0], 'b--', label='Zero Angle')
        self.ax3.plot(np.linspace(0,TIME_SIM,len(self.plot_activation_lta)), [MAX_ANGLE*1.1 if i else -MAX_ANGLE*1.1 for i in self.plot_activation_lta], label='LTA Activation', color = 'g')
        self.ax3.legend(loc='upper left')
        self.ax4.plot(np.linspace(0,TIME_SIM,len(self.plot_theta)), self.plot_theta, label='Theta')
        self.ax4.plot(np.linspace(0,TIME_SIM,len(self.plot_activation_lta)), [np.rad2deg(np.pi/2 - MAX_ANGLE)*1.1 if i else np.rad2deg(np.pi/2 + MAX_ANGLE)*0.9 for i in self.plot_activation_lta], label='LTA Activation', color = 'g')
        self.ax4.set_ylim(np.rad2deg(np.pi/2 + MAX_ANGLE),np.rad2deg(np.pi/2 - MAX_ANGLE))
        self.ax4.legend(loc='upper left')
        plt.show()
        
    def danger_zone(self, lane_width):
        theta = self.theta 
        car_position = [*self.car_position]
        corners = self.corners_car(car_position, just_front = True)
        
        if (self.distance_right[-1] < DISTANCE_THRESHOLD_LTA or self.distance_left[-1] < DISTANCE_THRESHOLD_LTA):
            self.use_lta = 1
            return True
        
        for _ in range(0, NUM_STEPS_LOOK_AHEAD):
            car_position[0] += self.velocity * np.cos(theta) * np.cos(self.phi) * DELTA_T * self.step_size_look_ahead
            car_position[1] += self.velocity * np.sin(theta) * np.cos(self.phi) * DELTA_T * self.step_size_look_ahead
            theta += self.velocity * np.sin(self.phi)/self.wheelbase * DELTA_T * self.step_size_look_ahead       
            corners = self.corners_car(car_position, just_front = True)
            
            if (corners[0][0] > lane_width or corners[1][0] < 0):
                self.use_lta = 1
                return True
            
        return False
class controller:
    def __init__(self):
        self.kp = 0.08 #0.08 
        self.ki = 0.01  #0.03 
        self.kd = 0.02 #0.035 
        self.I = 0
        self.error = 0
        self.reference = 0
        self.previous_error = 0
        
        

    def pid(self, current):
        error = self.reference - current        
        P = self.kp * error
        D = self.kd * (error - self.previous_error) / DELTA_T
        angle = P + self.I + D
        clip_angle = np.clip(angle, -MAX_ANGLE, MAX_ANGLE)
        self.I += self.ki * error * DELTA_T
        self.previous_error = error        
        return clip_angle
        
        
        
    
if __name__ == "__main__":
    car = Car(WHEELBASE, FRONT_WIDTH, WHEEL_RADIUS)        
    car.start_simulator()
