import curses
import matplotlib.pyplot as plt
import numpy as np
import time

# Constants
DELTA_T = 0.1  # Time interval in seconds
ANGLE_INCREMENT = np.radians(3)  # Angle increment in radians
MAX_ANGLE = np.radians(30)  # Maximum steering angle in radians
WHEELBASE = 2.36  # Wheelbase in meters
FRONT_WIDTH = 1.35 # Front wheel width in meters
WHEEL_RADIUS = 0.330  # Wheel radius in meters
LANE_WIDTH = 3.5  # Lane width in meters
FUTURE_LOOK_AHEAD = 2 # Future look ahead in seconds

class Car:
    def __init__(self, velocity, wheelbase, front_width, wheel_radius):
        self.car_position = (LANE_WIDTH/2, 0)  
        self.wheel_radius = wheel_radius
        self.front_width = front_width
    
        self.theta = np.radians(90)
        self.phi = 0
        self.velocity = velocity
        self.wheelbase = wheelbase
        self.look_ahead_iterations = int(FUTURE_LOOK_AHEAD/DELTA_T)
        
        plt.ion() 
        _, self.ax = plt.subplots()
        self.last_update_time = time.time() 
        self.warning_img = plt.imread('/home/rods/Desktop/IST-ROBO/Lab2/warning.png') 
    
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
                self.display_simulation()
        
    def joystick(self, stdscr):
        key = stdscr.getch()  
        if key == curses.KEY_LEFT:
            self.change_angle(ANGLE_INCREMENT)
            self.write_angles_screen(stdscr)
        elif key == curses.KEY_RIGHT:
            self.change_angle(-ANGLE_INCREMENT)
            self.write_angles_screen(stdscr)
        elif key == ord('x'):
            stdscr.addstr("Exiting...\n")
            plt.ioff()
            plt.close('all')
            return 1
        
        return 0
    
    def write_angles_screen(self, stdscr):
        stdscr.clear()
        stdscr.addstr(0, 0, f"The steering angle is {np.degrees(self.phi):=}")
        stdscr.refresh()    
    
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
    
    def display_simulation(self):
        self.ax.clear()
        x,y = self.car_position
        self.ax.scatter(x,y, label='Car Position', s= 30)
        dx, dy = self.get_direction()
        plt.arrow(x, y, dx*0.1, dy*0.1, head_width=0.2, head_length=0.1, fc='k', ec='k')
        self.ax.plot([0, 0], [-20, 20], 'k--', label='Lane Limit')
        self.ax.plot([LANE_WIDTH, LANE_WIDTH], [-20, 20], 'k--')
        if(self.danger_zone(LANE_WIDTH)):
            #Display the warning image
            self.ax.imshow(self.warning_img, extent=[x + 3.5, x + 4.5, y + 3.5 , y + 4.5]) 
            
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_xlim(x - 5, x + 5)
        self.ax.set_ylim(y - 5, y + 5)
        self.ax.legend(loc='upper left') 

        plt.draw()
        plt.pause(0.001)

if __name__ == "__main__":
    car = Car(4,WHEELBASE, FRONT_WIDTH, WHEEL_RADIUS)        
    curses.wrapper(car.start_simulator)
