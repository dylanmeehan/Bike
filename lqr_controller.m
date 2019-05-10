%Dylan Meehan (dem292@cornell.edu)
%March 2019

% get LQR controller for bicycle model

clear

% Define constants
   g = 9.81; %acceleration due to gravity
   l = 1.02; %length of wheel base 
   b = 0.33; %distance from rear wheel to COM projected onto ground
   h = 0.516; %height of COM in point mass model
    % h is not the same as the height of COM of the bicycle, h is
    % calculated to place the center of mass so that the point
    % mass model and the real bicycle fall with the same falling frequency.
    % see ABT Fall 2017 report for further discussion.
   c = 0;   %trail
 
%set speed to a constant: given with intial state in get_starting_state.py
v = 0.25;  %m/s
   
%Define system   
A = [   0       1       0
       g/h      0  -v^2/(h*l)
        0       0       0     ];
B = [   0  -b*v/(h*l)   1]';

%Define costs
Q = [1  0   0;
    0   0.05  0;
    0   0  0.05];
R = [.003];

%set timestep for LQR controller
Ts = 1/50;

%calculate LQR controller
[K,S,e] = lqrd(A,B,Q,R, Ts);
K = -1*K; %get K from lqr controller to match sign convention

disp("balance controller gains are")
vpa(K,8)