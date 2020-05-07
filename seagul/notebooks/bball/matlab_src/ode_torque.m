function dx = ode_torque(t,X,U)
params;

q1 = X(1); q2 = X(2);
dq1 = X(5); dq2 = X(6);

D = [[ M2*l1^2 + 2*M2*cos(q2)*l1*p2 + M1*p1^2 + M2*p2^2 + I1 + I2, M2*p2^2 + M2*l1*cos(q2)*p2 + I2]
[                            M2*p2^2 + M2*l1*cos(q2)*p2 + I2,                    M2*p2^2 + I2]];

C = [ - M2*l1*p2*sin(q2)*dq2^2 - 2*M2*dq1*l1*p2*sin(q2)*dq2 - M2*g*p2*sin(q1 + q2) - M2*g*l1*sin(q1) - M1*g*p1*sin(q1)
                                                                     -M2*p2*(- l1*sin(q2)*dq1^2 + g*sin(q1 + q2))];

d2x = D\(-C + U);

dx = [X(5:8); d2x; 0; -g];
                                                                 
% end

