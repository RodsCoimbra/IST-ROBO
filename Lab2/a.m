clear
close all
clc
v = 20;              % constant velocity
L = 2.36;              % wheelbase
Kp= 1.2;
Ki = 0.5;
Kd = 0.1;

% Open-loop transfer function G(s)
s = tf('s');
tf_Vehicle = v / (L * s);

% PID Controller C(s)
tf_PID = (Kp + Ki/s + Kd * s) ;

sys = tf_PID * tf_Vehicle;
% bode(sys)
% 
% figure
% nyqlog(sys)
% figure
b = feedback(sys, 1);
step(b)

