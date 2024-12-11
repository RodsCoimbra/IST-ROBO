import curses
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque

# Constants
DELTA_T = 0.1  # Time interval in seconds
ANGLE_INCREMENT = np.radians(5)  # Angle increment in radians
MAX_ANGLE = np.radians(30)  # Maximum steering angle in radians

class Car:
    def __init__(self, y, v):
        self.car_position = deque([y], maxlen=50)  # Keep only the last 50 measurements
        self.time = deque([0], maxlen=50)
        self.omega = 0
        self.v = v
        plt.ion() 
        _, self.ax = plt.subplots()
        self.last_update_time = time.time()  # Track last update time
    
    def joystick(self, stdscr):
        curses.cbreak()
        stdscr.nodelay(True)  # Make getch non-blocking
        
        while True:
            key = stdscr.getch()  # Get keyboard input (non-blocking)
            
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
                self.last_update_time = current_time
                self.time.append(self.time[-1] + DELTA_T)  # Simulate elapsed time
                self.car_position.append(self.omega)  # Assume no movement if no key press
                self.plot_car()
        
    def change_angle(self, angle):
        self.omega += angle
        self.omega = np.clip(self.omega, -MAX_ANGLE, MAX_ANGLE)
        
    def plot_car(self):
        # Clear and redraw the plot
        self.ax.clear()
        self.ax.plot(self.time, self.car_position, label='Steering Angle')
        self.ax.plot(self.time, np.ones(len(self.time)) * np.pi/8, label='Lane Limit Left')
        self.ax.plot(self.time, -np.ones(len(self.time)) * np.pi/8, label='Lane Limit Right')
        self.ax.set_xlabel('Time Step (s)')
        self.ax.set_ylabel('Steering Angle (radians)')
        self.ax.legend()
        # Update the plot
        plt.draw()
        plt.pause(0.001)

if __name__ == "__main__":
    car = Car(0, 0)        
    curses.wrapper(car.joystick)
