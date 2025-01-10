clear
close all
clc

DELTA_T = 0.01; 
kp= 0.2;
ki = 0.01 * DELTA_T;
kd = 0.03 / DELTA_T;
L = 2.36;
s = tf('s');

tf_PID = kp + ki/s + kd*s;
c = /s;
tf_Vehicle = cos(d)/s;
sys = tf_PID * tf_Vehicle;
bode(sys)

t = 0:0.01:10;       % Time vector from 0 to 10 seconds with 0.01s steps
omega = 10;           % Frequency of cosine wave (rad/s)
u = 3 * cos(omega * t);  % Input signal
[y, t_out] = lsim(sys, u, t);  % Simulate the response

plot(t, u, '--', 'DisplayName', 'Input (cos)');  % Plot input signal
hold on;
plot(t_out, y, 'DisplayName', 'System Response'); % Plot system response
xlabel('Time (s)');
ylabel('Amplitude');
legend;
grid on;

%%
t = deg2rad(-30:0.01:30);
y = tan(t);
plot(t,y);

eig