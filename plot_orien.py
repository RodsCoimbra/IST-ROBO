# %%
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

# %% [markdown]
# ### Funcs

# %%
def plot_trajectory(time, position):
    plt.figure(figsize=(10, 7))
    plt.plot(position[:, 0], position[:, 1], label="Trajectory")
    plt.scatter(position[0, 0], position[0, 1], color="green", label="Start", s=50)
    plt.scatter(position[-1, 0], position[-1, 1], color="red", label="End", s=50)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.title('Trajectory in XY Plane')
    plt.grid()
    plt.show()

    # Plot Z position over time to verify constant height
    plt.figure(figsize=(10, 5))
    plt.plot(time, position[:, 0], label="X Position", color='red')
    plt.plot(time, position[:, 1], label="Y Position", color='green')
    plt.plot(time, position[:, 2], label="Z Position", color='blue')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Positions Over Time')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot the trajectory
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(position[:, 0], position[:, 1], position[:, 2], label="Trajectory")
    ax.scatter(position[0, 0], position[0, 1], position[0, 2], color="green", label="Start", s=50)
    ax.scatter(position[-1, 0], position[-1, 1], position[-1, 2], color="red", label="End", s=50)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.legend()
    plt.show()
    
def trajectory(time, ang_vel, acceleration):
    """Ang vel need to be in rad/s"""
    diff_t = np.diff(time)
    orientations = []
    orientations.append(R.from_euler('xyz', [np.pi, 0, 0])) # It starts facing down (seen from the gravity acceleration from the accelerometer)

    for ang_v, dt in zip(ang_vel[1:], diff_t):
        ang_p = ang_v * dt
        delta_rot = R.from_euler('xyz', ang_p)
        orientations.append(orientations[-1] * delta_rot)

    position = [[0]*3]
    gravity = np.array([0, 0, -9.81])  # Gravity
    velocity = [0,0,0]
    
    for current_orientation, acc, dt in zip(orientations[1:], acceleration[1:], diff_t):
        # Rotate to world frame
        acc_world = current_orientation.as_matrix() @ acc
        # Subtract gravity
        acc_world = acc_world - gravity

        velocity = velocity + acc_world * dt
        position.append(position[- 1] + velocity * dt)
    
    plot_trajectory(time, np.array(position))   
    
def normalize_angle(angle):
    for i in range(len(angle[-1])):
        angle[-1][i] = (angle[-1][i] + 180) % 360 - 180
    return angle

def orientation_from_ang_vel(time, angular_vel):
    orientation = [[180,0,0]] #Initial orientation extracted by looking at the acceleration data (measuring the gravity acceleration at the beginning)
    normalize_angle(orientation)
    diff_t = np.diff(time)
    for ang_vel, dt in zip(angular_vel[1:], diff_t):
        orientation.append(orientation[-1] + ang_vel * dt)
        normalize_angle(orientation)

    orientation = np.radians(orientation)
    plt.figure(figsize=(15, 5))
    plt.plot(time, orientation)
    plt.xlabel('Time (s)')
    plt.ylabel('Orientation (deg)')
    plt.legend(['Roll', 'Pitch', 'Yaw'])
    plt.grid()
    
def plot_data(time, angular_vel, acceleration, title):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.title(title)
    plt.plot(time,angular_vel[:, 0], color='red')
    plt.plot(time,angular_vel[:, 1], color='green')
    plt.plot(time,angular_vel[:, 2], color = 'blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular velocity (deg/s)')
    plt.legend(['$w_X$', '$w_Y$', '$w_Z$'])
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(time,acceleration[:, 0], color='red')
    plt.plot(time,acceleration[:, 1], color='green')
    plt.plot(time,acceleration[:, 2], color = 'blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.legend(['$a_X$', '$a_Y$', '$a_Z$'])
    plt.grid()


import numpy as np

def plot_axes(ax, origin, roll, pitch, yaw):
    # Rotation matrices around x, y, and z axes
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = R_yaw @ R_pitch @ R_roll  # Note the order
    
    # Extract the rotated axes
    x_axis = R[:, 0] * 0.1  # Scale for visualization
    y_axis = R[:, 1] * 0.1
    z_axis = R[:, 2] * 0.1
    
    # Ensure axes are orthogonal
    assert np.allclose(np.dot(x_axis, y_axis), 0), "X and Y axes are not orthogonal"
    assert np.allclose(np.dot(x_axis, z_axis), 0), "X and Z axes are not orthogonal"
    assert np.allclose(np.dot(y_axis, z_axis), 0), "Y and Z axes are not orthogonal"
    
    # Plot the axes
    ax.quiver(*origin, *x_axis, color='r')  # X-axis
    ax.quiver(*origin, *y_axis, color='g')  # Y-axis
    ax.quiver(*origin, *z_axis, color='b')  # Z-axis

# %%
def dh_transform(a, alpha, d, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0,              np.sin(alpha),                 np.cos(alpha),                d],
        [0,              0,                             0,                            1]
    ])


def forward_kinematics(theta_list):
    T = np.eye(4)
    robot_dh = [[0.05,   np.pi/2, 0.3585],       # Base
                [0.3,    0,      -0.035],       # Shoulder
                [0.35,   0,       0],       # Elbow
                [0.251, -np.pi/2, 0],       # Wrist Pitch
                [0,      0,       0]]      # Wrist Roll
    joints = []
    for (theta, (a, alpha, d)) in zip(theta_list, robot_dh):
        T = np.dot(T, dh_transform(a, alpha, d, theta))
    joints.append(T[:3, 3])

    return joints


orientation = np.load('orientation.npy')
end_point = []
for i in range(orientation.shape[0]):
    joint_angles = [orientation[i,2] + np.pi/4, np.pi/2.4, -np.pi/1.6, orientation[i,1] + np.pi/4, orientation[i,0]]
    T = forward_kinematics(joint_angles)
    end_point.append(T[0])

end_point = np.array(end_point)



plt.figure(figsize=(10, 7))
plt.plot(end_point[:, 0], end_point[:, 1], label="End Effector Position")
plt.scatter(end_point[0, 0], end_point[0, 1], color="green", label="Start", s=50)
plt.scatter(end_point[-1, 0], end_point[-1, 1], color="red", label="End", s=50)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.legend()
plt.title('End Effector Position in XY Plane')
plt.grid()

# XZ Plane
plt.figure(figsize=(10, 7))
plt.plot(end_point[:, 0], end_point[:, 2], label="End Effector Position")
plt.scatter(end_point[0, 0], end_point[0, 2], color="green", label="Start", s=50)
plt.scatter(end_point[-1, 0], end_point[-1, 2], color="red", label="End", s=50)
plt.xlabel('X Position (m)')
plt.ylabel('Z Position (m)')
plt.legend()
plt.title('End Effector Position in XZ Plane')
plt.grid()

# YZ Plane
plt.figure(figsize=(10, 7))
plt.plot(end_point[:, 1], end_point[:, 2], label="End Effector Position")
plt.scatter(end_point[0, 1], end_point[0, 2], color="green", label="Start", s=50)
plt.scatter(end_point[-1, 1], end_point[-1, 2], color="red", label="End", s=50)
plt.xlabel('Y Position (m)')
plt.ylabel('Z Position (m)')
plt.legend()
plt.title('End Effector Position in YZ Plane')
plt.grid()


# Plot the end effector position in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(end_point[:, 0], end_point[:, 1], end_point[:, 2], label="End Effector Position")
ax.scatter(end_point[0, 0], end_point[0, 1], end_point[0, 2], color="green", label="Start", s=50)
plot_axes(ax, np.array(end_point[0]), orientation[0, 0], orientation[0, 1], orientation[0, 2])
ax.scatter(end_point[-1, 0], end_point[-1, 1], end_point[-1, 2], color="red", label="End", s=50)
plot_axes(ax, np.array(end_point[-1]), orientation[-1, 0], orientation[-1, 1], orientation[-1, 2])

# Plot the axes using quiver

ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')	
ax.set_zlabel('Z Position (m)')
# ax.set_xlim([0.2,1])
# ax.set_ylim([-0.8, 0])
ax.set_zlim([0.2, 1])
ax.legend()
plt.show()
