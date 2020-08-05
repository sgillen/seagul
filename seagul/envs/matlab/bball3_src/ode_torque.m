function dx = ode_torque(t,X,U)
params;

q1 = X(1); q2 = X(2); q3 = X(3);
dq1 = X(6); dq2 = X(7); dq3 = X(8);

D = [[ I1 + I2 + I3 + M2*l1^2 + M3*l1^2 + M3*l2^2 + M1*p1^2 + M2*p2^2 + M3*p3^2 + 2*M3*l1*p3*cos(q2 + q3) + 2*M3*l1*l2*cos(q2) + 2*M2*l1*p2*cos(q2) + 2*M3*l2*p3*cos(q3), M3*l2^2 + 2*M3*cos(q3)*l2*p3 + M3*l1*cos(q2)*l2 + M2*p2^2 + M2*l1*cos(q2)*p2 + M3*p3^2 + M3*l1*cos(q2 + q3)*p3 + I2 + I3, I3 + M3*p3^2 + M3*l1*p3*cos(q2 + q3) + M3*l2*p3*cos(q3)]
[                                          M3*l2^2 + 2*M3*cos(q3)*l2*p3 + M3*l1*cos(q2)*l2 + M2*p2^2 + M2*l1*cos(q2)*p2 + M3*p3^2 + M3*l1*cos(q2 + q3)*p3 + I2 + I3,                                                               M3*l2^2 + 2*M3*cos(q3)*l2*p3 + M2*p2^2 + M3*p3^2 + I2 + I3,                         M3*p3^2 + M3*l2*cos(q3)*p3 + I3]
[                                                                                                           I3 + M3*p3^2 + M3*l1*p3*cos(q2 + q3) + M3*l2*p3*cos(q3),                                                                                          M3*p3^2 + M3*l2*cos(q3)*p3 + I3,                                            M3*p3^2 + I3]];

C = [ - M3*g*l2*sin(q1 + q2) - M2*g*p2*sin(q1 + q2) - M2*g*l1*sin(q1) - M3*g*l1*sin(q1) - M1*g*p1*sin(q1) - M3*g*p3*sin(q1 + q2 + q3) - M3*dq2^2*l1*p3*sin(q2 + q3) - M3*dq3^2*l1*p3*sin(q2 + q3) - M3*dq2^2*l1*l2*sin(q2) - M2*dq2^2*l1*p2*sin(q2) - M3*dq3^2*l2*p3*sin(q3) - 2*M3*dq1*dq2*l1*p3*sin(q2 + q3) - 2*M3*dq1*dq3*l1*p3*sin(q2 + q3) - 2*M3*dq2*dq3*l1*p3*sin(q2 + q3) - 2*M3*dq1*dq2*l1*l2*sin(q2) - 2*M2*dq1*dq2*l1*p2*sin(q2) - 2*M3*dq1*dq3*l2*p3*sin(q3) - 2*M3*dq2*dq3*l2*p3*sin(q3)
                                                                                                                                                                                                                                                       M3*dq1^2*l1*p3*sin(q2 + q3) - M2*g*p2*sin(q1 + q2) - M3*g*p3*sin(q1 + q2 + q3) - M3*g*l2*sin(q1 + q2) + M3*dq1^2*l1*l2*sin(q2) + M2*dq1^2*l1*p2*sin(q2) - M3*dq3^2*l2*p3*sin(q3) - 2*M3*dq1*dq3*l2*p3*sin(q3) - 2*M3*dq2*dq3*l2*p3*sin(q3)
                                                                                                                                                                                                                                                                                                                                                                                 M3*p3*(dq1^2*l1*sin(q2 + q3) - g*sin(q1 + q2 + q3) + dq1^2*l2*sin(q3) + dq2^2*l2*sin(q3) + 2*dq1*dq2*l2*sin(q3))];

% U = [0; 0];
%  U = C;

d2x = D\(-C + U);

dx = [X(6:10); d2x; 0; -g];
                                                                 
% end

