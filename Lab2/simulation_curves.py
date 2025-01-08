import curses
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque 

# Constants
DELTA_T = 0.01  # Time interval in seconds
ANGLE_INCREMENT = np.radians(2)  # Angle increment in radians
MAX_ANGLE = np.radians(30)  # Maximum steering angle in radians
WHEELBASE = 2.36  # Wheelbase in meters
FRONT_WIDTH = 1.35 # Front wheel width in meters
WHEEL_RADIUS = 0.330  # Wheel radius in meters
LANE_WIDTH = 3.5  # Lane width in meters
MIN_VEL = 10
MAX_VEL = 30
FUTURE_LOOK_AHEAD = 0.5 # Future look ahead in seconds
NUM_STEPS_LOOK_AHEAD = 4
DISTANCE_THRESHOLD_LTA = 0.3
WINDOW_SIZE_MEAN = 10
LENGTH_CURVE = 32.4
NOISE_STD = 0.03
class Car:
    def __init__(self, velocity, wheelbase, front_width, wheel_radius):
        self.car_position = (LANE_WIDTH/2, -10)  
        self.wheel_radius = wheel_radius
        self.front_width = front_width
        self.distance_left = deque([self.car_position[0] - FRONT_WIDTH/2],maxlen=100)
        self.distance_right = deque([LANE_WIDTH - (self.car_position[0] + FRONT_WIDTH/2)],maxlen=100)
        self.theta = np.radians(90)
        self.phi = 0
        self.velocity = velocity
        self.wheelbase = wheelbase
        self.step_size_look_ahead = int(FUTURE_LOOK_AHEAD/(DELTA_T*NUM_STEPS_LOOK_AHEAD))
        self.controller = controller()
        self.use_lta = 0
        self.last_measurments = deque([0] * WINDOW_SIZE_MEAN,maxlen=WINDOW_SIZE_MEAN)
        self.num_seconds_sim = 0
        y_curve = np.linspace(0,LENGTH_CURVE, 1000) 
        self.y_trajectory = np.concatenate((np.linspace(-30,0,500), y_curve, np.linspace(LENGTH_CURVE, LENGTH_CURVE+30, 500)))
        self.x_trajectory = np.concatenate((np.zeros(500), 2 * np.sin(y_curve/5), np.zeros(500)))
        
        plt.ion() 
        _, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        self.last_update_time = 0
        self.last_update_time_sim = 0 
        
        self.warning_img = plt.imread('warning.png') 
    
    def start_simulator(self, stdscr):
        self.stdscr = stdscr
        curses.cbreak()
        self.stdscr.nodelay(True)  # Make getch non-blocking
        self.write_screen()
        while True:
            if(self.joystick()):
                break
            
            # Periodic update every Delta_T seconds
            current_time = time.time()
            
            if current_time - self.last_update_time >= DELTA_T:
                self.car_position = self.next_car_position()
                self.last_update_time = current_time
                self.corners = self.corners_car(self.car_position, self.theta)
                
                #Simulation update every 20*Delta_T seconds
                if current_time - self.last_update_time_sim >= 10* DELTA_T:
                    self.num_seconds_sim += 20*DELTA_T 
                    self.last_update_time_sim = current_time
                    self.display_simulation()
                    self.write_screen()
                    self.danger_zone()
                
                self.distance_to_lane()
                #Uses the LTA controller
                if self.use_lta:
                    predicted_error = self.distance_right[-1] - self.distance_left[-1] 
                    self.last_measurments.append(predicted_error)
                    predicted_error = np.mean(self.last_measurments)
                    desired_phi = self.controller.pid(predicted_error)
                    self.phi = desired_phi if abs(desired_phi - self.phi) <= ANGLE_INCREMENT else self.phi + np.sign(desired_phi - self.phi) * ANGLE_INCREMENT
        
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
        
    def next_car_position(self):
        dx, dy = self.get_direction()
        delta_x = dx * DELTA_T
        delta_y = dy * DELTA_T
        self.theta += self.velocity * np.tan(self.phi)/self.wheelbase * DELTA_T  
        x = self.car_position[0] + delta_x
        y = self.limit_inside_display(self.car_position[1] + delta_y, -20, LENGTH_CURVE+20)
        return (x,y)
    
    def limit_inside_display(self,value, min_value, max_value):
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
        
        if (self.distance_right[-1] < DISTANCE_THRESHOLD_LTA or self.distance_left[-1] < DISTANCE_THRESHOLD_LTA):
            self.use_lta = 1
            return
        
        theta = self.theta 
        car_position = [*self.car_position]
        for _ in range(0, NUM_STEPS_LOOK_AHEAD):
            car_position[0] += self.velocity * np.cos(theta) * DELTA_T * self.step_size_look_ahead
            car_position[1] += self.velocity * np.sin(theta) * DELTA_T * self.step_size_look_ahead
            theta += self.velocity * np.tan(self.phi)/self.wheelbase * DELTA_T * self.step_size_look_ahead       
            corners = self.corners_car(car_position, theta, just_front = True)
            
            idx1 = np.argmin(np.abs(self.y_trajectory - corners[0][1]))
            idx2 = np.argmin(np.abs(self.y_trajectory[idx1-5:idx1+5] - corners[1][1]))
            if (corners[0][0] > self.x_trajectory[idx1] + LANE_WIDTH or corners[1][0] < self.x_trajectory[idx2]):
                self.use_lta = 1
                return
        
        self.use_lta = 0
        return
    
    def corners_car(self, center_position, theta_initial, just_front = False):
        corners = []
        # Order - Front Right, Front Left, Back Left, Back Right
        angles = [theta_initial] if just_front else [theta_initial, theta_initial + np.pi]
        for theta in angles:
            x = center_position[0] +FRONT_WIDTH/2 * np.sin(theta) + WHEELBASE/2 * np.cos(theta)
            y = center_position[1] +WHEELBASE/2 * np.sin(theta) - FRONT_WIDTH/2 * np.cos(theta)
            corners.append(np.array([x, y]))
            x = center_position[0] -FRONT_WIDTH/2 * np.sin(theta) + WHEELBASE/2 * np.cos(theta)
            y = center_position[1] +WHEELBASE/2 * np.sin(theta) + FRONT_WIDTH/2 * np.cos(theta)
            corners.append(np.array([x, y]))
        
        return corners
    def distance_to_lane(self): 
        Front_right, Front_left = self.corners[0], self.corners[1]
        # LEFT
        idx = np.argmin(np.abs(self.y_trajectory - Front_left[1]))
        idx_min = max(idx - 5, 0)
        idx_max = min(idx + 5, len(self.x_trajectory))
        points = np.column_stack((self.x_trajectory[idx_min:idx_max], self.y_trajectory[idx_min:idx_max]))
        distance_left = (np.min(np.linalg.norm(points - Front_left, axis=1)))

        # RIGHT
        idx = np.argmin(np.abs(self.y_trajectory - Front_right[1]))
        idx_min = max(idx - 5, 0)
        idx_max = min(idx + 5, len(self.x_trajectory))
        points = np.column_stack((self.x_trajectory[idx_min:idx_max] + LANE_WIDTH, self.y_trajectory[idx_min:idx_max]))
        distance_right = (np.min(np.linalg.norm(points - Front_right, axis=1)))
    
        self.distance_left.append(distance_left + np.random.normal(0, NOISE_STD))
        self.distance_right.append(distance_right + np.random.normal(0, NOISE_STD))
        

    def display_simulation(self):
        self.ax1.clear()
        self.ax2.clear()
        x,y = self.car_position
        self.ax1.scatter(x,y, label='Car Position', s= 30)
        chassis_car = np.append(self.corners, [self.corners[0]], axis=0)
        self.ax1.plot(chassis_car[:,0],chassis_car[:,1])
        dx, dy = self.get_direction()
        self.ax1.arrow(x, y, dx*0.1, dy*0.1, head_width=0.2, head_length=0.1, fc='k', ec='k')
        #MAP LANE LIMITS
        self.ax1.plot(self.x_trajectory[0:-1:30], self.y_trajectory[0:-1:30], 'k--', label='Lane Limit')
        self.ax1.plot(self.x_trajectory[0:-1:30] + LANE_WIDTH, self.y_trajectory[0:-1:30], 'k--')
        
        if(self.use_lta):
            #Display the warning image
            self.ax1.imshow(self.warning_img, extent=[x + 3.5, x + 4.5, y + 3.5 , y + 4.5]) 
            
        self.ax1.set_xticklabels([])
        self.ax1.set_yticklabels([])
        self.ax1.set_xlim(x - 5, x + 5)
        self.ax1.set_ylim(y - 5, y + 5)
        self.ax1.legend(loc='upper left') 
        self.distance_to_lane()
        self.ax2.plot(np.arange(len(self.distance_left)), self.distance_left, label='Distance to left lane')
        self.ax2.plot(np.arange(len(self.distance_right)), self.distance_right, label='Distance to right lane')
        y = [i - j for i,j in zip(self.distance_right, self.distance_left)]
        self.ax2.plot(np.arange(len(y)), y, color= 'r', label='Distance to middle')
        self.ax2.legend(loc='upper left')
        self.ax2.set_ylim(-4, 4)
        
        plt.draw()
        plt.pause(0.001)
        
        
        
        

class controller:
    def __init__(self):
        self.kp = 0.2 #0.5
        self.ki = 0.01 #0.1
        self.kd = 0.03 #0.05
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
    car = Car(15,WHEELBASE, FRONT_WIDTH, WHEEL_RADIUS)        
    curses.wrapper(car.start_simulator)
