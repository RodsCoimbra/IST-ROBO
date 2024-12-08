clear
close all
% a alpha d theta
dhparams = [0.05   pi/2   0.3585   0;       % Base
            0.3   0      -0.035  0;       % Shoulder
            0.35    0     0          0;       % Elbow
            0.251  -pi/2  0         0;       % Wrist Pitch
            0     0      0    0];      % Wrist Roll

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

figure(Name=['Initial' num2str(i)])
show(robot)

figure(Name=['Moved' num2str(i)])
config = homeConfiguration(robot);
config(1).JointPosition = pi/4; 
config(2).JointPosition = pi/2.4;
config(3).JointPosition = -pi/1.6;
config(4).JointPosition = pi/4;
show(robot, config); 
xlim([-0.5, 1])
ylim([-0.5, 1])
zlim([0, 1])
