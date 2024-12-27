import curses
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque 

# Constants
DELTA_T = 0.01  # Time interval in seconds
ANGLE_INCREMENT = np.radians(3)  # Angle increment in radians
MAX_ANGLE = np.radians(30)  # Maximum steering angle in radians
WHEELBASE = 2.36  # Wheelbase in meters
FRONT_WIDTH = 1.35 # Front wheel width in meters
WHEEL_RADIUS = 0.330  # Wheel radius in meters
LANE_WIDTH = 3.5  # Lane width in meters
FUTURE_LOOK_AHEAD = 1.25 # Future look ahead in seconds
MIN_VEL = 10
MAX_VEL = 30
TIME_SIM = 20

class Car:
    def __init__(self, velocity, wheelbase, front_width, wheel_radius):
        self.car_position = (LANE_WIDTH/2, 0)  
        self.plot_car = []
        self.plot_error = []
        self.wheel_radius = wheel_radius
        self.front_width = front_width
        self.theta = np.radians(120)
        self.phi = 0
        self.plot_phi = []
        self.plot_theta = []
        self.velocity = velocity
        self.wheelbase = wheelbase
        self.look_ahead_iterations = int(FUTURE_LOOK_AHEAD/DELTA_T)
        self.controller = controller()
        _,(self.ax1, self.ax2) = plt.subplots(1,2)
        _, (self.ax3, self.ax4) = plt.subplots(1, 2)
    
    def start_simulator(self):
        i = 0
        while i < TIME_SIM/DELTA_T: #Run for 20 seconds
            i += 1
            self.car_position = self.next_car_position()
            self.plot_car.append(self.car_position)
            predicted_error = LANE_WIDTH/2 - self.car_position[0]
            self.phi = self.controller.pid(predicted_error)
            self.plot_phi.append(self.phi)
            self.plot_error.append(predicted_error)
        
        self.display_simulation()
        
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
        self.ax1.plot([0, 0], [0, 250], 'k--', label='Lane Limit')
        self.ax1.plot([LANE_WIDTH, LANE_WIDTH], [0, 250], 'k--')
        self.ax1.plot([LANE_WIDTH/2, LANE_WIDTH/2], [0, 250], 'b--', label='Path')
        self.ax1.legend(loc='upper left') 
        self.ax2.plot(np.linspace(0,20,len(self.plot_error)), self.plot_error, label='Error')
        self.ax3.scatter(np.linspace(0,20,len(self.plot_phi)), self.plot_phi, label='Phi Value', s= 10)
        self.ax3.plot([0,20], [MAX_ANGLE, MAX_ANGLE], 'r--', label='Max Angle')
        self.ax3.plot([0,20], [-MAX_ANGLE, -MAX_ANGLE], 'k--', label='Min Angle')
        self.ax3.plot([0,20], [0, 0], 'b--', label='Zero Angle')
        self.ax3.legend(loc='upper left')
        self.ax4.plot(np.linspace(0,20,len(self.plot_theta)), self.plot_theta, label='Theta')
        self.ax4.legend(loc='upper left')
        plt.show()

class controller:
    def __init__(self):
        self.kp = 0.06 #0.5
        self.ki = 0.0 #0.1
        self.kd = 0.05 #0.05
        self.ka = 0
        self.I = 0
        self.error = 0
        self.reference = 0
        self.previous_error = 0
        
        

    def pid(self, current):
        error = self.reference - current
        # If the error is too small, consider it as zero
        if error < 0.01 and error > -0.01:
            error = 0
            
        P = self.kp * error
        D = self.kd * (error - self.previous_error) / DELTA_T
        angle = P + self.I + D
        clip_angle = np.clip(angle, -MAX_ANGLE, MAX_ANGLE)
        
        #Anti wind-up
        self.I += (self.ki * error - self.ka * (clip_angle-angle)) * DELTA_T
        self.previous_error = error        
        return clip_angle
        
        
    
if __name__ == "__main__":
    car = Car(50,WHEELBASE, FRONT_WIDTH, WHEEL_RADIUS)        
    car.start_simulator()
