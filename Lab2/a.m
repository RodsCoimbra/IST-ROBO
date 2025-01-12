clear
close all
clc
v = 20;             
L = 2.36;             
Kp= 1.2;
Kd = 0.1;


s = tf('s');
tf_Vehicle = v / (L * s);

tf_PID = (Kp + Kd * s);

sys = tf_PID * tf_Vehicle;
bode(sys)

