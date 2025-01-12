import matplotlib.pyplot as plt
import numpy as np

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
FUTURE_LOOK_AHEAD = DISPLAY_DELTA       # Future look ahead in seconds
NUM_STEPS_LOOK_AHEAD = 4
DISTANCE_THRESHOLD_LTA = 1.5
WINDOW_SIZE_MEAN = 1
LENGTH_CURVE = 31.4
NOISE_STD = 0.00
SIM_TIME = 5

# Car wheels positions (Front Right, Front Left, Back Left, Back Right)
CAR_CORNERS = np.array([(WHEELBASE, -FRONT_WIDTH/2),
                        (WHEELBASE, FRONT_WIDTH/2),
                        (0, FRONT_WIDTH/2),
                        (0, -FRONT_WIDTH/2)]) 

SENSOR_POSITIONS = np.array([WHEELBASE, 0])

class Car:
    def __init__(self, velocity):
        # CAR Parameters
        self.car_position = (np.cos(0.3*np.pi) * 1.5, 0)  
        self.theta = np.radians(90)
        self.phi = np.radians(0)
        self.velocity = velocity
        
        # Simulation Parameters
        self.num_seconds_sim = 0
        self.last_update_time = 0
        self.last_update_time_sim = 0 
        
        # Distance to lane Parameters
        self.distance_left = []
        self.distance_right = []
        self.step_size_look_ahead = int(FUTURE_LOOK_AHEAD/(DELTA_T*NUM_STEPS_LOOK_AHEAD))
        self.dist_last_iteration = np.array([0,0])
        self.left_lane_point = None
        self.right_lane_point = None
        self.distance_error = []
        
        #MAP LANE LIMITS
        self.y_trajectory = np.linspace(0,200,2000)
        self.x_left_trajectory = np.zeros(2000)
        self.x_right_trajectory = self.x_left_trajectory + LANE_WIDTH
        
        # LTA Parameters
        self.use_lta = 0
        self.controller = controller(1.2, 0.0, 0.1) # PD Controller
        self.lta_activation = 0
        
        #Subplots and images 
        self.time = [0]
        self.plot_phi = [self.phi]
        self.plot_theta = [self.theta]
        self.plot_middle_point = []
        self.position_plot = [self.car_position]
        self.reference = []
        
    def start_simulator(self):
        self.sensor_position = rotation(SENSOR_POSITIONS, self.theta) + self.car_position
        self.distance_to_lane()
        while SIM_TIME > self.num_seconds_sim:
            self.num_seconds_sim += DELTA_T
            self.time.append(self.num_seconds_sim)
            self.car_position = self.next_car_position()

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
                
            self.distance_error.append(self.car_position[0])
            predicted_error = np.arctan2(self.middle_lane_point[1] - self.car_position[1], self.middle_lane_point[0] - self.car_position[0]) - self.theta
            self.phi = self.controller.pid(predicted_error, limits=(-MAX_ANGLE, MAX_ANGLE))
                    
        #self.display_final()
        np.save(f'Task5_results/car_position_{self.velocity}.npy', self.distance_error)

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
        #Limit theta between - pi and pi
        if self.theta > np.pi:
            self.theta -= 2*np.pi
        elif self.theta < -np.pi:
            self.theta += 2*np.pi
        
        x = self.car_position[0] + delta_x
        y = self.car_position[1] + delta_y
        return (x,y)
    
    def get_direction(self):
        dx = self.velocity * np.cos(self.theta)
        dy = self.velocity * np.sin(self.theta)
        return dx, dy
    
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
            
            start_idx_right = np.searchsorted(self.y_trajectory, corners[0][1], side="left")
            start_idx_right = np.argmin(np.abs(self.y_trajectory[max(0,start_idx_right-1):min(len(self.y_trajectory),start_idx_right+1)] - corners[0][1]))
            start_idx_left = np.argmin(np.abs(self.y_trajectory[max(0,start_idx_right-10):min(len(self.y_trajectory),start_idx_right+10)] - corners[1][1])) # -10 and +10 to be computationally faster, as both should have similar values
            if (corners[0][0] > self.x_right_trajectory[start_idx_right] or corners[1][0] < self.x_left_trajectory[start_idx_left]):
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
        
        right_lane_point, right_lane_dist = self.find_intersection(left_laser_angle, self.x_right_trajectory, self.y_trajectory)
        if right_lane_point is not None:
            self.right_lane_point = right_lane_point
            self.distance_right.append(right_lane_dist)
            self.dist_last_iteration[1] = 0
        
        
        if np.any(self.dist_last_iteration > TRUST_LAST_LTA_MEASUREMENT_TIME/DELTA_T):
            self.use_lta = -1
            return
        elif self.use_lta == -1:
            self.use_lta = 0
        
        self.middle_lane_point = [(self.left_lane_point[0] + self.right_lane_point[0])/2, (self.left_lane_point[1] + self.right_lane_point[1])/2]
        self.plot_middle_point.append(self.middle_lane_point)
        #self.distance_error.append(self.distance_right[-1] - self.distance_left[-1])
        
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
            
    # def display_final(self):
    #     #figures
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))  
    #     ax1.plot(self.time, np.degrees(self.plot_theta) - 90, label='Theta')
    #     ax1.set_xlabel('Time (s)')
    #     ax1.set_ylabel('Theta (degrees)')
    #     ax1.set_ylim(-60, 60)
    #     ax1.legend(loc='upper left')
    #     ax2.plot(self.time, np.degrees(self.plot_phi), label='Phi')
    #     max_angle = np.degrees(MAX_ANGLE)
    #     ax2.plot([0, self.time[-1]], [max_angle, max_angle], 'k--', label='Max Angle')
    #     ax2.plot([0, self.time[-1]], [-max_angle, -max_angle], 'k--', label='Min Angle')
    #     ax2.set_xlabel('Time (s)')
    #     ax2.set_ylabel('Phi (degrees)')
    #     ax2.legend(loc='upper left')
    #     ax1.grid()
    #     ax2.grid()
    #     fig.tight_layout()
        
    #     fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 8))  
    #     ax3.set_xlabel('X Position (m)')
    #     ax3.set_ylabel('Y Position (m)')
    #     self.create_corners_plot(ax3)
    #     position_plot = np.array(self.position_plot)
        
        
    #     ax3.scatter(*zip(*self.plot_middle_point[1:]), color='g', label='Reference Point')
    #     ax3.plot(self.x_left_trajectory[0:-1:30], self.y_trajectory[0:-1:30], 'k--', label='Lane Limit')
    #     ax3.plot(self.x_right_trajectory[0:-1:30], self.y_trajectory[0:-1:30], 'k--')
    #     ax3.legend(loc='upper left')
    #     ax3.set_xlim(LANE_WIDTH/2 - 10, LANE_WIDTH/2 + 10)
    #     ax3.set_ylim(position_plot[:,1].min() - 5, position_plot[:,1].max() + 5)
    #     ax4.plot(self.time[1:], self.distance_error, label="Car's Position")
    #     ax4.plot(self.time[1:], reference, 'g--', label='Reference')
    #     ax4.set_xlabel('Time (s)')
    #     ax4.set_ylabel('Distance (m)')
    #     ax4.legend(loc='upper left')
    #     ax4.set_xlim(self.time[1], self.time[-1])
    #     ax4.set_ylim(-10, 10)
    #     ax4.grid()
    #     fig2.tight_layout()
    #     plt.show()
        
       
    def create_corners_plot(self, ax3):
        for i in range(1, len(self.position_plot), 5):
            corners = self.corners_car(self.position_plot[i], self.plot_theta[i])
            chassis_car = np.append(corners, [corners[0]], axis=0)
            ax3.plot(chassis_car[:,0],chassis_car[:,1], color='blue', alpha=0.5)
            
        corners = self.corners_car(self.position_plot[i], self.plot_theta[i])
        chassis_car = np.append(corners, [corners[0]], axis=0)
        ax3.plot(chassis_car[:,0],chassis_car[:,1], color='blue', label="Car's Chassis", alpha=0.5)
              

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
    for vel in range(MIN_VEL, MAX_VEL + 1, 3):
        car = Car(vel)        
        car.start_simulator()
        del car


