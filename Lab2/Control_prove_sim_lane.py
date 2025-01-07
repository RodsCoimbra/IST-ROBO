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
LANE_WIDTH = 3.5  # Lane width in meters
MIN_VEL = 10
MAX_VEL = 33
NOISE_STD = 0.1
DISTANCE_THRESHOLD_LTA = 0.5
SIM_TIME = 10  # Simulation time in seconds
WINDOW_SIZE_MEAN = 20
FUTURE_LOOK_AHEAD = 2.0 # Future look ahead in seconds
NUM_STEPS_LOOK_AHEAD = 4
np.random.seed(42)

class Car:
    def __init__(self, velocity, wheelbase, front_width, wheel_radius):
        self.car_position = (LANE_WIDTH-1, 0)  
        self.wheel_radius = wheel_radius
        self.front_width = front_width
        self.theta = np.radians(90.5)
        self.phi = np.radians(0)
        self.velocity = velocity
        self.wheelbase = wheelbase
        self.num_seconds_sim = 0
        #plt.ion() 
        #_, (self.plt, self.ax2) = plt.subplots(1, 2, figsize=(15, 8))  
        self.step_size_look_ahead = int(FUTURE_LOOK_AHEAD/(DELTA_T*NUM_STEPS_LOOK_AHEAD))      
        self.theta_plot = [self.get_theta()]
        self.phi_plot = [self.get_phi()]
        self.last_update_time = 0
        self.last_update_time_sim = 0
        self.position_plot = [self.car_position]
        self.corners = self.corners_car(self.car_position)
        self.corners_plot_left = [self.corners[0]]
        self.corners_plot_right = [self.corners[1]]
        Front_right, Front_left = self.corners[0], self.corners[1]
        self.distance_left = [Front_left[0]]
        self.distance_right = [LANE_WIDTH - Front_right[0]]
        self.gt_distance_left = [Front_left[0]]
        self.gt_distance_right = [LANE_WIDTH - Front_right[0]]
        self.danger_plot = [False]
        
        self.distance_left_mean = deque([],maxlen=WINDOW_SIZE_MEAN)
        self.distance_right_mean = deque([],maxlen=WINDOW_SIZE_MEAN)
        self.time = [0]
    
    def start_simulator(self, stdscr):
        self.stdscr = stdscr
        curses.cbreak()
        self.stdscr.nodelay(True)  # Make getch non-blocking
        #self.write_screen()
        while self.num_seconds_sim < SIM_TIME:
            if(self.joystick()):
                break
            
            # current_time = time.time()
            # if current_time - self.last_update_time >= DELTA_T:
            self.num_seconds_sim += DELTA_T
            self.car_position = self.next_car_position()
            self.corners = self.corners_car(self.car_position)
            self.distance_to_lane_mean()
            
            #Simulation update every 20*Delta_T seconds
            if self.num_seconds_sim % (20 * DELTA_T):
                self.distance_to_lane()
                self.danger_plot.append(self.danger_zone())
                self.time.append(self.num_seconds_sim)
                self.theta_plot.append(self.get_theta())
                self.phi_plot.append(self.get_phi())
                self.position_plot.append(self.car_position)
                self.corners_plot_left.append(self.corners[0])
                self.corners_plot_right.append(self.corners[1])
                #self.display_simulation()
                    
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
        dx = self.velocity * np.cos(self.theta) * np.cos(self.phi)
        dy = self.velocity * np.sin(self.theta) * np.cos(self.phi)
        return dx, dy
    
    def next_car_position(self):
        dx, dy = self.get_direction()
        delta_x = dx * DELTA_T
        delta_y = dy * DELTA_T
        self.theta += self.velocity * np.sin(self.phi)/self.wheelbase * DELTA_T  
        x = self.car_position[0] + delta_x
        y = self.car_position[1] + delta_y
        return (x,y)
    
    def get_theta(self):
        theta = np.rad2deg(self.theta) - 90
        return theta % 360
    
    def get_phi(self):
        return np.rad2deg(self.phi)
     
    def distance_to_lane_mean(self):
        """"Only the front corners are considered"""
        Front_right, Front_left = self.corners[0], self.corners[1]
        self.distance_left_mean.append(Front_left[0] + np.random.normal(0, NOISE_STD))
        self.distance_right_mean.append(LANE_WIDTH - Front_right[0] + np.random.normal(0, NOISE_STD))
        
    def distance_to_lane(self):
        """"Only the front corners are considered"""
        Front_right, Front_left = self.corners[0], self.corners[1]
        self.distance_left.append(np.mean(self.distance_left_mean))
        self.distance_right.append(np.mean(self.distance_right_mean))
        self.gt_distance_left.append(Front_left[0])
        self.gt_distance_right.append(LANE_WIDTH - Front_right[0])
        
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

    def display_simulation(self):
        self.plt.clear()
        self.ax2.clear()
        x,y = self.car_position
        self.plt.scatter(x,y, label='Car Position', s= 30)
        chassis_car = np.append(self.corners, [self.corners[0]], axis=0)
        self.plt.plot(chassis_car[:,0],chassis_car[:,1])
        dx, dy = self.get_direction()
        self.plt.arrow(x, y, dx*0.1, dy*0.1, head_width=0.2, head_length=0.1, fc='k', ec='k')
        self.plt.plot([0, 0], [y - 10, y + 10], 'k--', label='Lane Limit')
        self.plt.plot([LANE_WIDTH, LANE_WIDTH], [y - 10, y + 10], 'k--')
        self.plt.set_xticklabels([])
        self.plt.set_xlim(x - 5, x + 5)
        self.plt.set_ylim(y - 5, y + 5)
        self.plt.legend(loc='upper left') 
        plt.draw()
        plt.pause(0.001)

    def danger_zone(self, lane_width = LANE_WIDTH):
        theta = self.theta 
        car_position = [*self.car_position]
        corners = self.corners_car(car_position, just_front = True)
        
        if (self.distance_right[-1] < DISTANCE_THRESHOLD_LTA or self.distance_left[-1] < DISTANCE_THRESHOLD_LTA):
            return True
        
        for _ in range(0, NUM_STEPS_LOOK_AHEAD):
            car_position[0] += self.velocity * np.cos(theta) * np.cos(self.phi) * DELTA_T * self.step_size_look_ahead
            car_position[1] += self.velocity * np.sin(theta) * np.cos(self.phi) * DELTA_T * self.step_size_look_ahead
            theta += self.velocity * np.sin(self.phi)/self.wheelbase * DELTA_T * self.step_size_look_ahead       
            corners = self.corners_car(car_position, just_front = True)
            
            if (corners[0][0] > lane_width or corners[1][0] < 0):
                return True
        
        return False
    
    def display_final(self):
        plt.ioff()
        plt.close('all')
        #figures
        plt.figure(figsize=(15, 8))
        
        plt.plot(self.time, self.distance_left, color=(0.85, 0.325, 0.098), label='Distance Left')
        plt.plot(self.time, self.distance_right, color=(0, 0.447, 0.741), label='Distance Right')
        plt.plot(self.time, self.gt_distance_left, color=(0.85, 0.325, 0.098), linestyle='--', label='Ground Truth Distance Left')
        plt.plot(self.time, self.gt_distance_right, color=(0, 0.447, 0.741), linestyle='--', label='Ground Truth Distance Right')
        plt.plot(np.linspace(0, SIM_TIME,len(self.danger_plot)), self.danger_plot, linestyle='--', label='LTA Activation', color = 'g')
        plt.plot([0, SIM_TIME], [DISTANCE_THRESHOLD_LTA, DISTANCE_THRESHOLD_LTA], 'k--', label='Activation Threshold')
        plt.grid()
        plt.tight_layout()
        plt.legend(loc='upper left')
        plt.xlabel('Time (s)')
        plt.ylabel('Distance (meters)')
        
        plt.figure(figsize=(15, 8))
        position_plot = np.array(self.position_plot)
        plt.plot(position_plot[:,0], position_plot[:,1], label='Center of the Car')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        corners_plot_left = np.array(self.corners_plot_left)
        corners_plot_right = np.array(self.corners_plot_right)
        plt.plot(corners_plot_left[:,0], corners_plot_left[:,1], label='Front Right Corner')
        plt.plot(corners_plot_right[:,0], corners_plot_right[:,1], label='Front Left Corner')
        plt.plot([0, 0], [position_plot[:,1].min(), position_plot[:,1].max()], 'k--', label='Lane Limit')
        plt.plot([LANE_WIDTH, LANE_WIDTH], [position_plot[:,1].min(), position_plot[:,1].max()], 'k--')
        plt.legend(loc='upper left')
        plt.xlim(LANE_WIDTH/2 - 5, LANE_WIDTH/2 + 5)
        plt.show()
        
    
if __name__ == "__main__":
    car = Car(18,WHEELBASE, FRONT_WIDTH, WHEEL_RADIUS)        
    curses.wrapper(car.start_simulator)
