# %%
import numpy as np
import matplotlib.pyplot as plt
# Data
positions = np.load("car_position.npy")
reference = np.load("reference.npy")
time = np.load("time.npy")

def steady_state_error(positions, reference):
    positions[-int(len(positions) * 0.1):]
    steady_state_value = np.mean(positions[-int(len(positions) * 0.1):])  # Last 10% of data
    steady_state_error = reference - steady_state_value
    print(f"Steady-State Error: {steady_state_error}")
    
def overshoot(positions, reference):
    peak = np.max(positions)
    overshoot = (peak - reference) / reference * 100
    print(f"Overshoot: {overshoot}%")
    
def settling_time(time, positions, reference, tolerance = 0.02):
    lower_bound = reference * (1 - tolerance)
    upper_bound = reference * (1 + tolerance)
    within_bounds = (positions >= lower_bound) & (positions <= upper_bound)
    settling_time_idx = np.where(~within_bounds)[0]
    settling_time = time[settling_time_idx[-1]+1]
    print(f"Settling Time: {settling_time} seconds")
        
def rise_time(time, positions, reference):
    t_10 = time[np.where(positions >= 0.1 * reference)[0][0]]
    t_90 = time[np.where(positions >= 0.9 * reference)[0][0]]
    rise_time = t_90 - t_10
    print(f"Rise Time: {rise_time} seconds")
    
MIN_VEL = 14
MAX_VEL = 32
# OPen the files in a Task5_results folder. Use Regex to extract the data. All have the same format, but depend on the velocity.
# Example: For 14m/s we have, Task5_results\car_position_14.npy and Task5_results\reference_14.npy 
time = np.load(f"Task5_results/time.npy")
time = time - time[0]
time = time[1:]
reference = 2
for vel in range(MIN_VEL, MAX_VEL + 1, 6):
    positions = np.load(f"Task5_results/car_position_{vel}.npy")
    print("-" * 20)
    print(f"Velocity: {vel} m/s")
    steady_state_error(positions, reference)
    overshoot(positions, reference)
    settling_time(time, positions, reference)
    rise_time(time, positions, reference)
    plt.plot(time, positions[1:], label=f"Velocity: {vel} m/s")
plt.title("Position vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.grid()
plt.legend()
plt.show()


