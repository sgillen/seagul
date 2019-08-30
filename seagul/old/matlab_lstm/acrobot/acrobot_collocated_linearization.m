function [dX,tau] = acrobot_collocated_linearization(t,X)
% dynamic equations of motion for acrobot, for use by ode45
% katiebyl 1/12/2010

% Define constants (geometry and mass properties):
m1=1; m2=1; L1=1; L2=1; Lc1=.5; Lc2=.5; I1=.2; I2=1; g=9.8;

% Extract state variables from X:
q1 = X(1); q2 = X(2); dq1 = X(3); dq2 = X(4);

% Need to define torque somehow...
tau = 0;
TAU = [0; tau];

m11 = m1*Lc1^2 + m2*(L1^2 + Lc2^2 + 2*L1*Lc2*cos(q2)) + I1 + I2;
m22 = m2*Lc2^2 + I2;
m12 = m2*(Lc2^2 + L1*Lc2*cos(q2)) + I2;
m21 = m12;
M = [m11, m12; m21, m22];

h1 = -m2*L1*Lc2*sin(q2)*dq2^2 - 2*m2*L1*Lc2*sin(q2)*dq2*dq1;
h2 = m2*L1*Lc2*sin(q2)*dq1^2;
H = [h1;h2];

phi1 = (m1*Lc1+m2*L1)*g*cos(q1) + m2*Lc2*g*cos(q1+q2);
phi2 = m2*Lc2*g*cos(q1+q2);
PHI = [phi1; phi2];

%---- All code ABOVE works for both collocated and non-collocated
%---- Some code BELOW must to altered for non-coll. control:

% Now, determine either q2des (coll.) or q1des (noncoll.)
alpha = 90*pi/180;
q2des = (2*alpha/pi)*atan(dq1);      % collocated, eqn 53

% For non-collocated control you may need to define
% a different set of intermediate variables here:
T = [-(m11^-1)*(m12); 1];            % eqn 10
M22bar = T'*M*T;                     % collocated
h2bar = h2 - m21*(m11^-1)*h1;        % collocated
phi2bar = phi2 - m21*(m11^-1)*phi1;  % collocated

% Control gains and control law to set tau:
kp=50; kd=5; % gains not selected with care!
% For non-collocated PFL, replace next two lines, using eqns 56 and 35
v2 = kp*(q2des - q2) - kd*dq2;  % collocated, eqn 55 (Spong94)
tau = M22bar*v2 + h2bar + phi2bar; % collocated, eqn 11

% Set torque_limit below to limit actuator torque:
torque_limit = 100000;  % [Nm] limit in torque magnitude
tau = sign(tau)*min(abs(tau),torque_limit);
TAU = [0; tau];


%---- Code below works for both collocated and non-collocated:
% Finally, calculate dX, the function output:
% M*d2Q + H + PHI = TAU
% So,
% d2Q = inv(M)*(TAU - H - PHI);

d2Q = (M^-1)*(TAU - H - PHI);
dX = [dq1; dq2; d2Q];


