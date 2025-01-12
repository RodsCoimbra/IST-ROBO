import curses
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque 

# Constants
DELTA_T = 0.05                          # Time interval in seconds
DISPLAY_DELTA = 2 * DELTA_T             # Display update interval in seconds
TRUST_LAST_LTA_MEASUREMENT_TIME = 1     # Time in seconds to trust the last LTA measurement
ANGLE_INCREMENT = np.radians(2)         # Angle increment in radians
MAX_ANGLE = np.radians(20)              # Maximum steering angle in radians
WHEELBASE = 2.36                        # Wheelbase in meters
FRONT_WIDTH = 1.35                      # Front wheel width in meters
LANE_WIDTH = 4                          # Lane width in meters
MIN_VEL = 14
MAX_VEL = 32
FUTURE_LOOK_AHEAD = 3*DISPLAY_DELTA       # Future look ahead in seconds
NUM_STEPS_LOOK_AHEAD = 4
DISTANCE_THRESHOLD_LTA = 2
WINDOW_SIZE_MEAN = 1
LENGTH_CURVE = np.pi * 10
LENGTH_STRAIGHT = 60
NOISE_STD = 0.00
SIM_TIME = 6

# Car wheels positions (Front Right, Front Left, Back Left, Back Right)
CAR_CORNERS = np.array([(WHEELBASE, -FRONT_WIDTH/2),
                        (WHEELBASE, FRONT_WIDTH/2),
                        (0, FRONT_WIDTH/2),
                        (0, -FRONT_WIDTH/2)]) 

SENSOR_POSITIONS = np.array([WHEELBASE, 0])

class Car:
    def __init__(self, velocity):
        # CAR Parameters
        self.car_position = (LANE_WIDTH/2, 0)  
        self.theta = np.radians(90)
        self.phi = 0
        self.velocity = velocity
        
        # Simulation Parameters
        self.num_seconds_sim = 0
        self.last_update_time = 0
        self.last_update_time_sim = 0 
        
        # Distance to lane Parameters
        self.distance_left = deque([],maxlen=100)
        self.distance_right = deque([],maxlen=100)
        self.step_size_look_ahead = int(FUTURE_LOOK_AHEAD/(DELTA_T*NUM_STEPS_LOOK_AHEAD))
        self.dist_last_iteration = np.array([0,0])
        self.left_lane_point = None
        self.right_lane_point = None
        
        #MAP LANE LIMITS
        y_curve = np.linspace(0,LENGTH_CURVE, 1000) 
        self.y_trajectory = np.concatenate((np.linspace(0,LENGTH_STRAIGHT,LENGTH_STRAIGHT*10), y_curve + LENGTH_STRAIGHT, np.linspace(LENGTH_CURVE+LENGTH_STRAIGHT, LENGTH_CURVE+2*LENGTH_STRAIGHT, LENGTH_STRAIGHT*10)))
        self.x_left_trajectory = np.concatenate((np.zeros(LENGTH_STRAIGHT*10), 2 * np.sin(y_curve/10), np.zeros(LENGTH_STRAIGHT*10)))
        self.x_right_trajectory = self.x_left_trajectory + LANE_WIDTH
        
        # LTA Parameters
        self.use_lta = 0
        self.lta_activation = 0
        self.controller = controller(1.2, 0.0, 0.1) # PD Controller
        #Subplots and images 
        plt.ion() 
        _, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 8))
        self.time = [0]
        self.plot_phi = [self.phi]
        self.plot_theta = [self.theta]
        self.plot_middle_point = []
        self.position_plot = [self.car_position]
        self.plot_lta = [0]
        self.plot_distance_left = []
        self.plot_distance_right = []
        self.warning_img = plt.imread('warning.png') 
        self.lta_working_img = plt.imread('lta_working.webp')
        self.unavailable = plt.imread('unavailable.png')
    
    def start_simulator(self, stdscr):
        self.stdscr = stdscr
        curses.cbreak()
        self.stdscr.nodelay(True)  # Make getch non-blocking
        while SIM_TIME > self.num_seconds_sim:
            if(self.joystick()):
                break
            
            current_time = time.time()
            
            if current_time - self.last_update_time >= DELTA_T:
                self.num_seconds_sim += DELTA_T 
                self.time.append(self.num_seconds_sim)
                self.car_position = self.next_car_position()
                self.last_update_time = current_time
                self.corners = self.corners_car(self.car_position, self.theta)
                self.sensor_position = rotation(SENSOR_POSITIONS, self.theta) + self.car_position
                self.distance_to_lane()
                self.danger_zone()  
                self.plot_theta.append(self.theta)
                self.plot_phi.append(self.phi)
                self.position_plot.append(self.car_position)
                
                #LTA controller
                #Error is the orientation from car position to middle lane point - Current orientation
                if self.middle_lane_point[1] < self.car_position[1]:
                    self.use_lta = -1
                predicted_error = np.arctan2(self.middle_lane_point[1] - self.car_position[1], self.middle_lane_point[0] - self.car_position[0]) - self.theta
                self.plot_lta.append(self.use_lta)
                if self.use_lta > 0:
                    self.phi = self.controller.pid(predicted_error, limits=(-MAX_ANGLE, MAX_ANGLE))
                    
                else:
                    self.controller.reset(predicted_error)
                     
                    
                #Simulation update every DISPLAY_DELTA seconds
                if current_time - self.last_update_time_sim >= DISPLAY_DELTA:
                    self.last_update_time_sim = current_time
                    self.display_simulation()
                    
        self.display_final()
                        
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
    
    def change_angle(self, angle_vel):
        self.phi += angle_vel
        self.phi = np.clip(self.phi, -MAX_ANGLE, MAX_ANGLE)
        
    def next_car_position(self):
        dx, dy = self.get_direction()
        delta_x = dx * DELTA_T
        delta_y = dy * DELTA_T
        self.theta += self.velocity * np.tan(self.phi)/WHEELBASE * DELTA_T  
        #Limit theta between - pi and pu
        if self.theta > np.pi:
            self.theta -= 2*np.pi
        elif self.theta < -np.pi:
            self.theta += 2*np.pi
        
        x = self.car_position[0] + delta_x
        y = self.car_position[1] + delta_y
        #y = self.limit_inside_display(self.car_position[1] + delta_y)
        return (x,y)
    
    def get_direction(self):
        dx = self.velocity * np.cos(self.theta)
        dy = self.velocity * np.sin(self.theta)
        return dx, dy
    
    def find_closest_point_car(self, y_trajectory, corner):
        start_idx_right = np.searchsorted(y_trajectory, corner, side="left") - 1
        #Ensure that the idxs are within the trajectory array
        idx_min = max(0,start_idx_right) 
        idx_max = min(len(y_trajectory),start_idx_right+2)
        # Find the closest point to the car between the two values
        closest_point = np.abs(y_trajectory[idx_min:idx_max] - corner)
        return idx_min + np.argmin(closest_point)
    
    
    def danger_zone(self):
        if self.use_lta == -1:
            return
        
        if (self.distance_right[-1] < DISTANCE_THRESHOLD_LTA or self.distance_left[-1] < DISTANCE_THRESHOLD_LTA):
            self.use_lta = 2
            self.lta_activation = 0
            return
        
        theta = self.theta 
        car_position = [*self.car_position]
        for _ in range(0, NUM_STEPS_LOOK_AHEAD):
            car_position[0] += self.velocity * np.cos(theta) * DELTA_T * self.step_size_look_ahead
            car_position[1] = car_position[1] + self.velocity * np.sin(theta) * DELTA_T * self.step_size_look_ahead #self.limit_inside_display(car_position[1] + self.velocity * np.sin(theta) * DELTA_T * self.step_size_look_ahead)
            theta += self.velocity * np.tan(self.phi)/WHEELBASE * DELTA_T * self.step_size_look_ahead       
            corners = self.corners_car(car_position, theta, just_front = True)
            
            closest_point_right = self.find_closest_point_car(self.y_trajectory, self.corners[0][1])
            closest_point_left = self.find_closest_point_car(self.y_trajectory, self.corners[1][1])
            if (corners[0][0] > self.x_right_trajectory[closest_point_right] or corners[1][0] < self.x_left_trajectory[closest_point_left]):
                self.use_lta = 1
                self.lta_activation = 0
                return
        
        self.lta_activation += 1
        if self.lta_activation > 5:
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
        self.dist_last_iteration+=1

        left_lane_point, left_lane_dist = self.find_intersection(right_laser_angle, self.x_left_trajectory, self.y_trajectory)
        if left_lane_point is not None:     
            self.left_lane_point = left_lane_point
            self.distance_left.append(left_lane_dist)
            self.dist_last_iteration[0] = 0
            self.plot_distance_left.append(left_lane_dist)
        else:
            self.plot_distance_left.append(-1)
        
        right_lane_point, right_lane_dist = self.find_intersection(left_laser_angle, self.x_right_trajectory, self.y_trajectory)
        if right_lane_point is not None:
            self.right_lane_point = right_lane_point
            self.distance_right.append(right_lane_dist)
            self.dist_last_iteration[1] = 0
            self.plot_distance_right.append(right_lane_dist)
        else:
            self.plot_distance_right.append(-1)
        
        
        if np.any(self.dist_last_iteration > TRUST_LAST_LTA_MEASUREMENT_TIME/DELTA_T):
            self.use_lta = -1
            return
        elif self.use_lta == -1:
            self.use_lta = 0
        
        self.middle_lane_point = (self.left_lane_point[0] + self.right_lane_point[0])/2, (self.left_lane_point[1] + self.right_lane_point[1])/2
        self.plot_middle_point.append(self.middle_lane_point)
        
    def display_simulation(self):
        self.ax1.clear()
        self.ax2.clear()
        
        # Car position, orientation and chassis
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
            if self.dist_last_iteration[1] == 0:
                self.ax1.plot([self.sensor_position[0], self.right_lane_point[0]], [self.sensor_position[1], self.right_lane_point[1]], 'b--')
            if self.dist_last_iteration[0] == 0:
                self.ax1.plot([self.sensor_position[0], self.left_lane_point[0]], [self.sensor_position[1], self.left_lane_point[1]], 'b--')
            self.ax1.plot([self.left_lane_point[0], self.right_lane_point[0]], [self.left_lane_point[1], self.right_lane_point[1]], 'r--')
            self.ax1.scatter(*self.middle_lane_point, s=50, color='r', label='Reference Point')
        
        #LTA information
        self.img_display_lta(x, y)
             
        # Data
        data = f"Velocity: {self.velocity}\nSteering Angle: {round(np.degrees(self.phi),3)}\n Time: {round(self.num_seconds_sim,2)}"
        self.ax1.text(
            0.05, 0.05,        
            data,
            transform=self.ax1.transAxes,
            horizontalalignment='left',
            verticalalignment='bottom'
        )
        
        # Distance to lane
        self.ax2.plot(np.arange(len(self.distance_left)), self.distance_left, label='Distance to left lane')
        self.ax2.plot(np.arange(len(self.distance_right)), self.distance_right, label='Distance to right lane')

        # Parameters for subplots
        #self.ax1.set_xticklabels([])
        self.ax1.set_yticklabels([])
        self.ax1.set_xlim(x - 5, x + 5)
        self.ax1.set_ylim(y - 2.5, y + 7.5)
        self.ax1.legend(loc='upper left')
        self.ax1.set_xlabel('X Position (m)')
         
        
        self.ax2.legend(loc='upper left')
        self.ax2.set_ylim(-5, 8)
        
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
            
    def display_final(self):
        plt.ioff()
        plt.close('all')
        #figures
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))  
        lta_was_active = np.array([1 if x > 0 else 0 for x in self.plot_lta])
        ax1.plot(self.time, np.degrees(self.plot_theta) - 90, label='Theta')
        ax1.plot(self.time, 20*lta_was_active, 'g--', label='LTA Activation')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Theta (degrees)')
        ax1.set_ylim(-60, 60)
        ax1.legend(loc='upper left')
        ax2.plot(self.time, np.degrees(self.plot_phi), label='Phi')
        ax2.plot(self.time, 10*lta_was_active, 'g--', label='LTA Activation')
        max_angle = np.degrees(MAX_ANGLE)
        ax2.plot([0, self.time[-1]], [max_angle, max_angle], 'k--', label='Max Angle')
        ax2.plot([0, self.time[-1]], [-max_angle, -max_angle], 'k--', label='Min Angle')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Phi (degrees)')
        ax2.legend(loc='upper left')
        fig.tight_layout()
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 8))
        
        ax3.plot(self.time, self.plot_lta, label='LTA', color='#FFA500')
        ax3.set_ylabel('LTA Status')
        ax3.set_xlabel('Time (s)')
        #horizontal line at 0
        ax3.plot([0, self.time[-1]], [0, 0], 'k--', label='No LTA')
        #horizontal line at 1
        ax3.plot([0, self.time[-1]], [1, 1], 'r--', label='Predictive LTA')
        #horizontal line at 2
        ax3.plot([0, self.time[-1]], [2, 2], 'b--', label='Distance Threshold LTA')
        #horizontal line at -1
        ax3.plot([0, self.time[-1]], [-1, -1], 'g--', label='LTA Unavailable')
        ax3.set_xlabel('Time (s)')
        ax3.legend(loc='upper left')
        
        ax4.plot(self.time[1:], self.plot_distance_left, label='Distance to left lane')
        ax4.plot(self.time[1:], self.plot_distance_right, label='Distance to right lane')
        ax4.plot([self.time[1], self.time[-1]], [DISTANCE_THRESHOLD_LTA, DISTANCE_THRESHOLD_LTA], 'k--', label='Distance Threshold')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Distance (m)')
        ax4.set_ylim(-2, 8)
        ax4.legend(loc='upper left')
        fig2.tight_layout()
        
        plt.figure(figsize=(15, 8))
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        self.create_corners_plot()
        position_plot = np.array(self.position_plot)
        
        x_vals = [p[0] for p in self.plot_middle_point[0:-1:2]]
        y_vals = [p[1] for p in self.plot_middle_point[0:-1:2]]
        colors = ['g' if x > 0 else 'r' for x in self.plot_lta[1:-1:2]]

        # Separate points by color
        green_points = [(x, y) for (x, y, c) in zip(x_vals, y_vals, colors) if c == 'g']
        red_points   = [(x, y) for (x, y, c) in zip(x_vals, y_vals, colors) if c == 'r']

        # Plot two scatter sets
        if green_points:
            plt.scatter(*zip(*green_points), color='g', label='Reference Point (LTA Active)')
        if red_points:
            plt.scatter(*zip(*red_points), color='r', label='Reference Point (LTA Deactivated)')
        
        plt.plot(self.x_left_trajectory[0:-1:30], self.y_trajectory[0:-1:30], 'k--', label='Lane Limit')
        plt.plot(self.x_right_trajectory[0:-1:30], self.y_trajectory[0:-1:30], 'k--')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.xlim(LANE_WIDTH/2 - 10, LANE_WIDTH/2 + 10)
        plt.ylim(position_plot[:,1].min() - 5, position_plot[:,1].max() + 5)
        plt.show()
       
    def create_corners_plot(self):
        for i in range(1, len(self.position_plot), 5):
            corners = self.corners_car(self.position_plot[i], self.plot_theta[i])
            chassis_car = np.append(corners, [corners[0]], axis=0)
            plt.plot(chassis_car[:,0],chassis_car[:,1], color='blue', alpha=0.5)
            
        corners = self.corners_car(self.position_plot[i], self.plot_theta[i])
        chassis_car = np.append(corners, [corners[0]], axis=0)
        plt.plot(chassis_car[:,0],chassis_car[:,1], color='blue', label="Car's Chassis", alpha=0.5)
              

class controller:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.I = 0
        self.error = 0
        self.reference = 0
        self.previous_error = 0
        
    def pid(self, error, limits = None):
        P = self.kp * error
        D = self.kd * (error - self.previous_error) 
        output = P + self.I + D
        if limits is not None:
            output = np.clip(output, limits[0], limits[1])
        self.I += self.ki * error 
        self.previous_error = error        
        return output
    
    def reset(self, error):
        self.I = 0
        self.previous_error = error
       
       
def rotation(point, theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ point
    
    
if __name__ == "__main__":
    car = Car(20)        
    curses.wrapper(car.start_simulator)


