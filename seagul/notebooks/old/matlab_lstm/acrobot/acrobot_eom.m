function dX = acrobot_eom(t,X)
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
M = [m11, m12; m12, m22];

h1 = -m2*L1*Lc2*sin(q2)*dq2^2 - 2*m2*L1*Lc2*sin(q2)*dq2*dq1;
h2 = m2*L1*Lc2*sin(q2)*dq1^2;
H = [h1;h2];

phi1 = (m1*Lc1+m2*L1)*g*cos(q1) + m2*Lc2*g*cos(q1+q2);
phi2 = m2*Lc2*g*cos(q1+q2);
PHI = [phi1; phi2];

% M*d2Q + H + PHI = TAU
% d2Q = inv(M)*(TAU - H - PHI);

TAU;

-H;

-PHI;

d2Q = (M^-1)*(TAU - H - PHI);
dX = [dq1; dq2; d2Q];


