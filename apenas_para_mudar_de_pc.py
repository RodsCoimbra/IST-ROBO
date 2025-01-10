import curses
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque 

# Constants
DELTA_T = 0.01  # Time interval in seconds
ANGLE_INCREMENT = np.radians(1.5)  # Angle increment in radians
MAX_ANGLE = np.radians(30)  # Maximum steering angle in radians
WHEELBASE = 2.36  # Wheelbase in meters
FRONT_WIDTH = 1.35 # Front wheel width in meters
WHEEL_RADIUS = 0.330  # Wheel radius in meters
LANE_WIDTH = 4  # Lane width in meters
MIN_VEL = 10
MAX_VEL = 33
NOISE_STD = 0.0
SIM_TIME = 3  # Simulation time in seconds

class Car:
    def __init__(self, velocity, wheelbase, front_width, wheel_radius):
        self.car_position = (LANE_WIDTH/2, 0)  
        self.wheel_radius = wheel_radius
        self.front_width = front_width
        self.theta = np.radians(95)
        self.phi = np.radians(0)
        self.velocity = velocity
        self.wheelbase = wheelbase
        self.num_seconds_sim = 0
        plt.ion() 
        _, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))       
        self.theta_plot = [self.get_theta()]
        self.theta_list = [self.theta]
        self.phi_plot = [self.get_phi()]
        self.last_update_time = 0
        self.last_update_time_sim = 0
        self.position_plot = [self.car_position]
        self.corners = self.corners_car(self.car_position, self.theta)
        self.corners_plot_left = [self.corners[0]]
        self.corners_plot_right = [self.corners[1]]
        self.time = [0]
    
    def start_simulator(self, stdscr):
        self.stdscr = stdscr
        curses.cbreak()
        self.stdscr.nodelay(True)  # Make getch non-blocking
        self.write_screen()
        while self.num_seconds_sim < SIM_TIME:
            if(self.joystick()):
                break
            
            current_time = time.time()
            if current_time - self.last_update_time >= DELTA_T:
                self.num_seconds_sim += DELTA_T
                self.car_position = self.next_car_position()
                self.last_update_time = current_time
                self.corners = self.corners_car(self.car_position, self.theta)
                
                
                #Simulation update every 15*Delta_T seconds
                if current_time - self.last_update_time_sim >= 10* DELTA_T:
                    self.time.append(self.num_seconds_sim)
                    self.last_update_time_sim = current_time
                    self.theta_plot.append(self.get_theta())
                    self.theta_list.append(self.theta)
                    self.phi_plot.append(self.get_phi())
                    self.position_plot.append(self.car_position)
                    self.display_simulation()
                    
        self.display_final()
    def joystick(self):
        key = self.stdscr.getch()  
        if key == curses.KEY_LEFT:
            self.change_angle(ANGLE_INCREMENT)
            self.write_screen()
        elif key == curses.KEY_RIGHT:
            self.change_angle(-ANGLE_INCREMENT)
            self.write_screen()
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
    
    def write_screen(self):
        self.stdscr.clear()
        self.stdscr.addstr(0, 0, f"The steering angle is {round(-np.degrees(self.phi),3):=}")
        self.stdscr.addstr(1, 0, f"The velocity is {round(self.velocity,0):=}")
        self.stdscr.addstr(2, 0, f"The simulation time is {self.num_seconds_sim:=}")
        self.stdscr.refresh()    
    
    def change_velocity(self, increment):
        self.velocity += increment
        self.velocity = np.clip(self.velocity, MIN_VEL, MAX_VEL)
    
    def change_angle(self, angle):
        self.phi += angle
        self.phi = np.clip(self.phi, -MAX_ANGLE, MAX_ANGLE)
        
    def get_direction(self):
        dx = self.velocity * np.cos(self.theta)
        dy = self.velocity * np.sin(self.theta)
        return dx, dy
    
    def next_car_position(self):
        dx, dy = self.get_direction()
        delta_x = dx * DELTA_T
        delta_y = dy * DELTA_T
        self.theta += self.velocity * np.tan(self.phi)/self.wheelbase * DELTA_T  
        x = self.car_position[0] + delta_x
        y = self.car_position[1] + delta_y
        return (x,y)
    
    def get_theta(self):
        theta = np.rad2deg(self.theta) - 90
        return theta % 360
    
    def get_phi(self):
        return np.rad2deg(self.phi)
     
    def corners_car(self, center_position, theta_val, just_front = False):
        corners = []
        # Order - Front Right, Front Left, Back Left, Back Right
        angles = [theta_val] if just_front else [theta_val, theta_val + np.pi]
        for theta in angles:
            x = center_position[0] +FRONT_WIDTH/2 * np.sin(theta) + WHEELBASE/2 * np.cos(theta)
            y = center_position[1] +WHEELBASE/2 * np.sin(theta) - FRONT_WIDTH/2 * np.cos(theta)
            corners.append(np.array([x, y]))
            x = center_position[0] -FRONT_WIDTH/2 * np.sin(theta) + WHEELBASE/2 * np.cos(theta)
            y = center_position[1] +WHEELBASE/2 * np.sin(theta) + FRONT_WIDTH/2 * np.cos(theta)
            corners.append(np.array([x, y]))
        
        return corners

    def display_simulation(self):
        self.ax1.clear()
        self.ax2.clear()
        x,y = self.car_position
        self.ax1.scatter(x,y, label='Car Position', s= 30)
        chassis_car = np.append(self.corners, [self.corners[0]], axis=0)
        self.ax1.plot(chassis_car[:,0],chassis_car[:,1])
        dx, dy = self.get_direction()
        self.ax1.arrow(x, y, dx*0.1, dy*0.1, head_width=0.2, head_length=0.1, fc='k', ec='k')
        self.ax1.plot([0, 0], [y - 10, y + 10], 'k--', label='Lane Limit')
        self.ax1.plot([LANE_WIDTH, LANE_WIDTH], [y - 10, y + 10], 'k--')
        self.ax1.set_xticklabels([])
        self.ax1.set_xlim(x - 5, x + 5)
        self.ax1.set_ylim(y - 5, y + 5)
        self.ax1.legend(loc='upper left') 
        plt.draw()
        plt.pause(0.001)

    def display_final(self):
        plt.ioff()
        plt.close('all')
        #figures
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))  
        
        ax1.plot(self.time, self.theta_plot)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Theta (degrees)')
        ax2.plot(self.time, self.phi_plot)
        ax2.plot([0, self.time[-1]], [30, 30], 'k--', label='Max Angle')
        ax2.plot([0, self.time[-1]], [-30, -30], 'k--', label='Min Angle')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Phi (degrees)')
        
        plt.figure(figsize=(15, 8))
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        self.create_corners_plot()
        position_plot = np.array(self.position_plot)
        
        plt.plot( label='Front Right Corner')
        plt.plot( label='Front Left Corner')
        plt.plot([0, 0], [position_plot[:,1].min(), position_plot[:,1].max()], 'k--', label='Lane Limit')
        plt.plot([LANE_WIDTH, LANE_WIDTH], [position_plot[:,1].min(), position_plot[:,1].max()], 'k--')
        plt.legend(loc='upper left')
        plt.xlim(LANE_WIDTH/2 - 5, LANE_WIDTH/2 + 5)
        # plt.ylim(5, 15)
        plt.show()
        
    def create_corners_plot(self):
        for i in range(1, len(self.position_plot), 5):
            corners = self.corners_car(self.position_plot[i], self.theta_list[i])
            chassis_car = np.append(corners, [corners[0]], axis=0)
            plt.plot(chassis_car[:,0],chassis_car[:,1], color='blue')
            
        corners = self.corners_car(self.position_plot[i], self.theta_list[i])
        chassis_car = np.append(corners, [corners[0]], axis=0)
        plt.plot(chassis_car[:,0],chassis_car[:,1], color='blue', label="Car's Chassis")
if __name__ == "__main__":
    car = Car(20,WHEELBASE, FRONT_WIDTH, WHEEL_RADIUS)        
    curses.wrapper(car.start_simulator)
