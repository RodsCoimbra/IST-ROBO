clc
clear
syms x y theta phi V omega_s L real
syms dx dy dtheta dphi dV domega_s real

% Define the state and input variables
state = [x; y; theta; phi];
input = [V; omega_s];

% Define the system dynamics
f = [V*cos(theta); 
     V*sin(theta); 
     V*tan(phi)/L;
     omega_s];

% Linearize the system: Compute the Jacobians
A = jacobian(f, state);
B = jacobian(f, input);

% Substitute the nominal operating point (theta = 0, phi = 0, V = V0)
theta_eq = 0; phi_eq = 0; V_eq = V; % Assuming constant velocity
A_lin = simplify(subs(A, [theta, phi, V], [theta_eq, phi_eq, V_eq]));
B_lin = simplify(subs(B, [theta, phi, V], [theta_eq, phi_eq, V_eq]));

% Display the results
%disp('Linearized A matrix:');
%+
% disp(A_lin);
% disp('Linearized B matrix:');
% disp(f)
% disp(B);
% disp(B_lin);

%%
clc
clear
close all
% Define parameters
v = 20;              % constant velocity
L = 2.36;              % wheelbase
Kp= 1.8;
Ki = 0.001;
Kd = 0.08;

% Open-loop transfer function G(s)
s = tf('s');
G = v / (L * s);

% PID Controller C(s)
C = Kp + Ki/s + Kd * s;

LTA = C *G;

%Bode Plot
figure;
bode(LTA);
title('Bode Diagram of Open-Loop System');

figure;
sys = feedback(LTA, 1);
step(sys)
figure;
t = 0:0.01:2*pi;       % Time vector from 0 to 10 seconds with 0.01s steps
omega = 1;           % Frequency of cosine wave (rad/s)
u = 3 * cos(omega * t);  % Input signal
[y, t_out] = lsim(sys, u, t);  % Simulate the response

plot(t, u, '--', 'DisplayName', 'Input (cos)');  % Plot input signal
hold on;
plot(t_out, y, 'DisplayName', 'System Response'); % Plot system response
xlabel('Time (s)');
ylabel('Amplitude');
legend;
grid on;
% Nyquist plot
% figure;
% nyquist(C * G);
% title('Nyquist Plot of Open-Loop System');
% figure;
% rlocus(C*G)

% controlSystemDesigner(G,C)