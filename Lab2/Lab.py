import curses
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque 

# Constants
DELTA_T = 0.1  # Time interval in seconds
ANGLE_INCREMENT = np.radians(3)  # Angle increment in radians
MAX_ANGLE = np.radians(30)  # Maximum steering angle in radians
WHEELBASE = 2.36  # Wheelbase in meters
FRONT_WIDTH = 1.35 # Front wheel width in meters
WHEEL_RADIUS = 0.330  # Wheel radius in meters
LANE_WIDTH = 10.5  # Lane width in meters
FUTURE_LOOK_AHEAD = 2 # Future look ahead in seconds
MIN_VEL = 0
MAX_VEL = 15

class Car:
    def __init__(self, velocity, wheelbase, front_width, wheel_radius):
        self.car_position = (LANE_WIDTH/2, 0)  
        self.wheel_radius = wheel_radius
        self.front_width = front_width
        self.time = deque([0],maxlen=100)
        self.distance_left = deque([self.car_position[0] - FRONT_WIDTH/2],maxlen=100)
        self.distance_right = deque([LANE_WIDTH - (self.car_position[0] + FRONT_WIDTH/2)],maxlen=100)
        self.theta = np.radians(90)
        self.phi = 0
        self.velocity = velocity
        self.wheelbase = wheelbase
        self.look_ahead_iterations = int(FUTURE_LOOK_AHEAD/DELTA_T)
        
        plt.ion() 
        _, (self.ax1, self.ax2) = plt.subplots(2, 1)
        self.last_update_time = time.time() 
        
        self.warning_img = plt.imread('warning.png') 
    
    def start_simulator(self, stdscr):
        curses.cbreak()
        stdscr.nodelay(True)  # Make getch non-blocking
        self.write_angles_screen(stdscr)
        while True:
            if(self.joystick(stdscr)):
                break

            # Periodic update every Delta_T seconds
            current_time = time.time()
            if current_time - self.last_update_time >= DELTA_T:
                self.car_position = self.next_car_position()
                self.last_update_time = current_time
                self.time.append(self.time[-1] + DELTA_T)
                self.display_simulation()
        
    def joystick(self, stdscr):
        key = stdscr.getch()  
        if key == curses.KEY_LEFT:
            self.change_angle(ANGLE_INCREMENT)
            self.write_angles_screen(stdscr)
        elif key == curses.KEY_RIGHT:
            self.change_angle(-ANGLE_INCREMENT)
            self.write_angles_screen(stdscr)
        elif key == curses.KEY_UP:
            self.change_velocity(3)
        elif key == curses.KEY_DOWN:
            self.change_velocity(-3)
        elif key == ord('x'):
            stdscr.addstr("Exiting...\n")
            plt.ioff()
            plt.close('all')
            return 1
        
        return 0
    
    def write_angles_screen(self, stdscr):
        stdscr.clear()
        stdscr.addstr(0, 0, f"The steering angle is {round(-np.degrees(self.phi),0):=}")
        stdscr.addstr(1, 0, f"The velocity is {round(self.velocity,0):=}")
        stdscr.refresh()    
    
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
        self.theta += self.velocity * np.sin(self.phi)/self.wheelbase * DELTA_T  
        x = self.car_position[0] + delta_x
        y = self.limit_inside_display(self.car_position[1] + delta_y, -10, 10)
        return (x,y)
    
    def get_direction(self):
        dx = self.velocity * np.cos(self.theta) * np.cos(self.phi)
        dy = self.velocity * np.sin(self.theta) * np.cos(self.phi)
        return dx, dy
    
    def limit_inside_display(self,value, min_value, max_value):
        if value > max_value:
            value = min_value + (value - max_value)
        elif value < min_value:
            value = max_value - (min_value - value)
        
        return value
    
    def danger_zone(self, lane_width):
        theta = self.theta 
        x_position = self.car_position[0]
        for _ in range(self.look_ahead_iterations):
            x_position += self.velocity * np.cos(theta) * np.cos(self.phi)* DELTA_T
            theta += self.velocity * np.sin(self.phi)/self.wheelbase * DELTA_T
            if (x_position < 0 or x_position > lane_width):
                return True
        
        return False
    
    def display_corners_car(self):
        self.corners = []
        # Order - Front Right, Front Left, Back Left, Back Right
        for theta in [self.theta, self.theta + np.pi]:
            x = self.car_position[0] +FRONT_WIDTH/2 * np.sin(theta) + WHEELBASE/2 * np.cos(theta)
            y = self.car_position[1] +WHEELBASE/2 * np.sin(theta) - FRONT_WIDTH/2 * np.cos(theta)
            self.corners.append(np.array([x, y]))
            x = self.car_position[0] -FRONT_WIDTH/2 * np.sin(theta) + WHEELBASE/2 * np.cos(theta)
            y = self.car_position[1] +WHEELBASE/2 * np.sin(theta) + FRONT_WIDTH/2 * np.cos(theta)
            self.corners.append(np.array([x, y]))
    
    def distance_to_lane(self):
        """"Only the front corners are considered"""
        Front_right, Front_left = self.corners[0], self.corners[1]
        if (Front_right - Front_left)[0] < 0:
            Front_right, Front_left = Front_left, Front_right
        self.distance_left.append(Front_left[0])
        self.distance_right.append(LANE_WIDTH - Front_right[0])

    def display_simulation(self):
        self.ax1.clear()
        self.ax2.clear()
        x,y = self.car_position
        self.ax1.scatter(x,y, label='Car Position', s= 30)
        self.display_corners_car()
        chassis_car = np.append(self.corners, [self.corners[0]], axis=0)
        self.ax1.plot(chassis_car[:,0],chassis_car[:,1])
        dx, dy = self.get_direction()
        self.ax1.arrow(x, y, dx*0.1, dy*0.1, head_width=0.2, head_length=0.1, fc='k', ec='k')
        self.ax1.plot([0, 0], [-20, 20], 'k--', label='Lane Limit')
        self.ax1.plot([LANE_WIDTH, LANE_WIDTH], [-20, 20], 'k--')
        if(self.danger_zone(LANE_WIDTH)):
            #Display the warning image
            self.ax1.imshow(self.warning_img, extent=[x + 3.5, x + 4.5, y + 3.5 , y + 4.5]) 
            
        self.ax1.set_xticklabels([])
        self.ax1.set_yticklabels([])
        self.ax1.set_xlim(x - 5, x + 5)
        self.ax1.set_ylim(y - 5, y + 5)
        self.ax1.legend(loc='upper left') 
        self.distance_to_lane()
        self.ax2.plot(self.time, self.distance_left, label='Distance to left lane')
        self.ax2.plot(self.time, self.distance_right, label='Distance to right lane')

        plt.draw()
        plt.pause(0.001)

    
if __name__ == "__main__":
    car = Car(4,WHEELBASE, FRONT_WIDTH, WHEEL_RADIUS)        
    curses.wrapper(car.start_simulator)
