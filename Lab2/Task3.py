import curses
import matplotlib.pyplot as plt
import numpy as np
import time

DELTA_T = 0.02  # Time interval in seconds
DISPLAY_DELTA = 5 * DELTA_T # Display update interval in seconds
ANGLE_INCREMENT = np.radians(2)  # Angle increment in radians
MAX_ANGLE = np.radians(20)  # Maximum steering angle in radians
WHEELBASE = 2.36  # Wheelbase in meters
FRONT_WIDTH = 1.35 # Front wheel width in meters
LANE_WIDTH = 4  # Lane width in meters
MIN_VEL = 14
MAX_VEL = 32
SIM_TIME = 7  # Simulation time in seconds
LENGTH_CURVE = 62.8

CAR_CORNERS = np.array([(WHEELBASE, -FRONT_WIDTH/2),
                        (WHEELBASE, FRONT_WIDTH/2),
                        (0, FRONT_WIDTH/2),
                        (0, -FRONT_WIDTH/2)]) 

class Car:
    def __init__(self, velocity):
        # CAR Parameters
        self.car_position = (LANE_WIDTH/2, -35)  
        self.theta = np.radians(90)
        self.phi = 0
        self.velocity = velocity
        
        # Simulation Parameters
        self.num_seconds_sim = 0
        self.last_update_time = 0
        self.last_update_time_sim = 0
        
        #Map
        y_curve = np.linspace(0,LENGTH_CURVE, 5000) 
        self.y_trajectory = np.concatenate((np.linspace(-45,0,450), y_curve, np.linspace(LENGTH_CURVE, LENGTH_CURVE+45, 450)))
        self.x_left_trajectory = np.concatenate((np.zeros(450), 2 * np.sin(y_curve/10), np.zeros(450)))
        self.x_right_trajectory = self.x_left_trajectory + LANE_WIDTH
        
        
        #Subplots 
        plt.ion() 
        _, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 8))       
        self.theta_plot = [self.get_theta()]
        self.theta_list = [self.theta]
        self.phi_plot = [self.get_phi()]
        self.position_plot = [self.car_position]
        self.corners = self.corners_car(self.car_position, self.theta)
        self.time = [0]
    
    def start_simulator(self, stdscr):
        self.stdscr = stdscr
        curses.cbreak()
        self.stdscr.nodelay(True)  # Make getch non-blocking
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
                if current_time - self.last_update_time_sim >= DISPLAY_DELTA:
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
        self.theta += self.velocity * np.tan(self.phi)/WHEELBASE * DELTA_T  
        x = self.car_position[0] + delta_x
        y = self.car_position[1] + delta_y
        return (x,y)
    
    def get_theta(self):
        theta = np.rad2deg(self.theta) - 90
        return theta % 360
    
    def get_phi(self):
        return np.rad2deg(self.phi)
     
    def corners_car(self, car_position, theta, just_front = False):
        corners = []
        num_corners = 2 if just_front else 4
        for i in range(num_corners):
            x, y = rotation(CAR_CORNERS[i], theta) + car_position
            corners.append([x,y])
            
        return np.array(corners)

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
        
        # Data
        data = f"Velocity: {self.velocity}\nSteering Angle: {round(np.degrees(self.phi),3)}"
        self.ax1.text(
            0.025, 0.025,        
            data,
            transform=self.ax1.transAxes,
            horizontalalignment='left',
            verticalalignment='bottom'
        )
        
        # Parameters for subplots
        self.ax1.set_yticklabels([])
        self.ax1.set_xlabel('X Position (m)')
        self.ax1.set_xlim(x - 5, x + 5)
        self.ax1.set_ylim(y - 2.5, y + 7.5)
        self.ax1.legend(loc='upper left') 
        plt.draw()
        plt.pause(0.001)

    def display_final(self):
        plt.ioff()
        plt.close('all')
        #figures
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))  
        
        ax1.plot(self.time, self.theta_plot)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Theta (degrees)')
        ax2.plot(self.time, self.phi_plot)
        max_angle = np.degrees(MAX_ANGLE)
        ax2.plot([0, self.time[-1]], [max_angle, max_angle], 'k--', label='Max Angle')
        ax2.plot([0, self.time[-1]], [-max_angle, -max_angle], 'k--', label='Min Angle')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Phi (degrees)')
        fig.tight_layout()
        
        plt.figure(figsize=(15, 8))
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        self.create_corners_plot()
        position_plot = np.array(self.position_plot)
        
        plt.plot(self.x_left_trajectory[0:-1:30], self.y_trajectory[0:-1:30], 'k--', label='Lane Limit')
        plt.plot(self.x_right_trajectory[0:-1:30], self.y_trajectory[0:-1:30], 'k--')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.xlim(LANE_WIDTH/2 - 10, LANE_WIDTH/2 + 10)
        plt.ylim(position_plot[:,1].min() - 5, position_plot[:,1].max() + 5)
        plt.show()
       
    def create_corners_plot(self):
        for i in range(1, len(self.position_plot), 4):
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
