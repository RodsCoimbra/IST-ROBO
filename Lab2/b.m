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
theta_r = 0;        % assume straight-line motion (cos(theta_r) = 1)
DELTA_T = 0.01; 
Kp= 0.2;
Ki = 0.01 * DELTA_T;
Kd = 0.03 / DELTA_T;

% Open-loop transfer function G(s)
s = tf('s');
G = (v^2 * cos(theta_r)) / (L * s^2);

% PID Controller C(s)
C = Kp + Ki/s + Kd * s

% Bode Plot
% figure;
% bode(C * G);
% title('Bode Diagram of Open-Loop System');
% 
% % Nyquist plot
% figure;
% nyquist(C * G);
% title('Nyquist Plot of Open-Loop System');
% figure;
% rlocus(C*G)

%controlSystemDesigner(G, C)

E = (8.8845*(s+0.06616)*(s+0.0005038))/s

