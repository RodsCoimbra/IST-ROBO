clear
close all
% dhparams = [0.05   pi/2   0.3585    0;       % Base
%             0.3    0      -0.035    0;       % Shoulder
%             0.35    0      0         0;       % Elbow
%             0.251   0   0           0;       % Wrist Pitch
%             0.1      -pi/2      0    0];      % Wrist Roll

dhparams = [0.05   pi/2   0.3585   0;       % Base
            0.3   0      -0.035  0;       % Shoulder
            0.35    0     0          0;       % Elbow
            0.251  -pi/2  0         0;       % Wrist Pitch
            0     0      0    0];      % Wrist Roll


% a alpha d theta

% red - x, y - green, blue z
robot = rigidBodyTree;
num_joints = 5;
bodies = cell(num_joints,1);
joints = cell(num_joints,1);
for i = 1:num_joints
    bodies{i} = rigidBody(['body' num2str(i)]);
    joints{i} = rigidBodyJoint(['jnt' num2str(i)],"revolute");
    setFixedTransform(joints{i},dhparams(i,:),"dh");
    bodies{i}.Joint = joints{i};
    if i == 1 % Add first body to base
        addBody(robot,bodies{i},"base")
    else % Add current body to previous body by name
        addBody(robot,bodies{i},bodies{i-1}.Name)
    end
end

% show(robot);                

for i = num_joints:-1:1
    figure(Name=['Initial' num2str(i)])
    config = homeConfiguration(robot); % Retrieve current joint configurations
    config(i).JointPosition = 0; % Set last joint to rotate by pi/4
    show(robot, config); % Display the robot with the updated configuration



    figure(Name=['Moved' num2str(i)])
    config = homeConfiguration(robot); % Retrieve current joint configurations
    config(i).JointPosition = pi/4; % Set last joint to rotate by pi/4
    show(robot, config); % Display the robot with the updated configuration
end
