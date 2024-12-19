import curses
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque

# Constants
DELTA_T = 0.1  # Time interval in seconds
ANGLE_INCREMENT = np.radians(6)  # Angle increment in radians
MAX_ANGLE = np.radians(30)  # Maximum steering angle in radians
WHEELBASE = 2.36  # Wheelbase in meters
FRONT_WIDTH = 1.35 # Front wheel width in meters
WHEEL_RADIUS = 0.330  # Wheel radius in meters

class Car:
    def __init__(self, initial_position, velocity, wheelbase, front_width, wheel_radius):
        self.car_position = deque([[initial_position, initial_position]], 30)  # Keep only the last 50 measurements
        #self.time = deque([0], 50)
        self.wheel_radius = wheel_radius
        self.front_width = front_width
    
        self.steering_angle = 0
        self.theta = 0
        self.phi = 0
        self.velocity = velocity
        self.wheelbase = wheelbase
        
        plt.ion() 
        _, self.ax = plt.subplots()
        self.last_update_time = time.time()  
    
    def joystick(self, stdscr):
        curses.cbreak()
        stdscr.nodelay(True)  # Make getch non-blocking
        
        while True:
            key = stdscr.getch()  
            if key == curses.KEY_LEFT:
                self.change_angle(ANGLE_INCREMENT)
            elif key == curses.KEY_RIGHT:
                self.change_angle(-ANGLE_INCREMENT)
            elif key == ord('x'):
                stdscr.addstr("Exiting...\n")
                plt.ioff()
                plt.close('all')
                break
            
            # Periodic update every 0.1 seconds
            current_time = time.time()
            if current_time - self.last_update_time >= DELTA_T:
                self.update_car_position()
                self.last_update_time = current_time
                self.plot_car()
        
    def change_angle(self, angle):
        last_steering_angle = self.steering_angle
        self.steering_angle += angle
        self.steering_angle = np.clip(self.steering_angle, -MAX_ANGLE, MAX_ANGLE)
        self.phi += self.steering_angle - last_steering_angle
                
    def update_car_position(self):
        delta_x = self.velocity * np.cos(self.theta) * np.cos(self.phi)* DELTA_T
        delta_y = self.velocity * np.sin(self.theta) * np.cos(self.phi) * DELTA_T
        self.theta += self.velocity * np.sin(self.phi)/self.wheelbase * DELTA_T  
        self.car_position.append([self.car_position[-1][0] + delta_x, self.car_position[-1][1] + delta_y])
    
    def plot_car(self):
        self.ax.clear()
        car = np.array(self.car_position)
        self.ax.plot(car[:,0], car[:,1], label='Car Position')
        self.ax.scatter(self.car_position[-1][0], self.car_position[-1][1], c='r', label='Current Position')
        # Create a square of 100x100 meters
        self.ax.plot([-50, 50, 50, -50, -50], [-50, -50, 50, 50, -50], 'k--', label='Lane Limit')
        # self.ax.plot(np.ones(len(self.time)) * 3.5,self.time, 'k--', label='Lane Limit')
        # self.ax.plot(np.ones(len(self.time)) * -3.5, self.time, 'k--')
        self.ax.set_ylabel('Time Step (s)')
        self.ax.set_xlabel('Steering Angle (radians)')
        self.ax.legend()
        # Update the plot
        plt.draw()
        plt.pause(0.001)

if __name__ == "__main__":
    car = Car(0, 10,WHEELBASE, FRONT_WIDTH, WHEEL_RADIUS)        
    curses.wrapper(car.joystick)
