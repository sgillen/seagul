
%% Function to calculate velocities post impact
function Xpost = impact(X)

% linkComX = rand(10,1)

q1 = X(1); q2 = X(2); q3 = X(3); xb = X(4); yb = X(5);
dq1 = X(6); dq2 = X(7); dq3 = X(8); vbx = X(9); vby = X(10);

th1 = q1;
th2 = q1 + q2;
th3 = q1 + q2 + q3;
thetaPerp = th3 + pi/2;

params;

x1 =  -l1*sin(th1);
x1cm = -p1*sin(th1);
x2 = x1 - l2*sin(th2);
x2cm = x1 - p2*sin(th2);
x3 = x2 - l3*sin(th3);
x3cm = x2 - p3*sin(th3);

y1 = l1*cos(th1);
y1cm = p1*cos(th1);
y2 = y1 + l2*cos(th2);
y2cm = y1 + p2*cos(th2);
y3 = y2 + l3*cos(th3);
y3cm = y2 + p3*cos(th3);


if (x3 == x2)
    disp('link 3 in vertical position');
    Xpost = zeros(10,1);
    return
end

ImpLinkSlope = (y3 - y2)/(x3 - x2);
impLinkIntercept = y3 - ImpLinkSlope*x3;

if ImpLinkSlope < 1e-10
    ximp = xb;
    yimp = y3;
    limp = abs(xb - x2);
else
    perpLineSlope = -1/ImpLinkSlope;
    perpLineIntercept = yb - perpLineSlope*xb;

    interceptPoint = [-ImpLinkSlope 1; -perpLineSlope 1]\[impLinkIntercept; perpLineIntercept];

    limp = sqrt((interceptPoint(1) - x2)^2 + (interceptPoint(2) - y2)^2);

    ximp = x2 - limp*sin(th3);
    yimp = y2 + limp*cos(th3);
end

mass = [M1,M2,M3];
inertia = [I1,I2,I2];
linkComX = [x1cm,x2cm,x3cm];
linkComY = [y1cm,y2cm,y3cm];

Mx = zeros(3,3);
My = zeros(3,3);
Mt = zeros(3,3);

% Conserving angular momentum about hinge
xp = 0;
yp = 0;

index = [1,2,3];
for n=1:length(index)
    Mx(1,index(n)) = -mass(index(n))*[linkComY(index(n)) - yp];
    My(1,index(n)) = mass(index(n))*[linkComX(index(n)) - xp];
    Mt(1,index(n)) = inertia(index(n));
    
end

% Conserving angular momentum about joint 1

xp = x1;
yp = y1;

index = [2,3];

for n=1:length(index)
    Mx(2,index(n)) = -mass(index(n))*[linkComY(index(n)) - yp];
    My(2,index(n)) = mass(index(n))*[linkComX(index(n)) - xp];
    Mt(2,index(n)) = inertia(index(n));    
end

% Conserving angular momentum about joint 2

xp = x2;
yp = y2;

index = [3];

for n=1:length(index)
    Mx(3,index(n)) = -mass(index(n))*[linkComY(index(n)) - yp];
    My(3,index(n)) = mass(index(n))*[linkComX(index(n)) - xp];
    Mt(3,index(n)) = inertia(index(n));    
end

% Now the Jacobians

JX = [[                                           -p1*cos(q1),                                        0,                     0]
[                        - p2*cos(q1 + q2) - l1*cos(q1),                         -p2*cos(q1 + q2),                     0]
[ - l2*cos(q1 + q2) - l1*cos(q1) - p3*cos(q1 + q2 + q3), - l2*cos(q1 + q2) - p3*cos(q1 + q2 + q3), -p3*cos(q1 + q2 + q3)]];

JY = [[                                           -p1*sin(q1),                                        0,                     0]
[                        - p2*sin(q1 + q2) - l1*sin(q1),                         -p2*sin(q1 + q2),                     0]
[ - l2*sin(q1 + q2) - l1*sin(q1) - p3*sin(q1 + q2 + q3), - l2*sin(q1 + q2) - p3*sin(q1 + q2 + q3), -p3*sin(q1 + q2 + q3)]];

Jth = [1 0 0; 1 1 0; 1 1 1];


armAMmass = Mx*JX + My*JY + Mt*Jth;
totalAMmass = [armAMmass(1,1) armAMmass(1,2) armAMmass(1,3) -Mb*yimp Mb*ximp
           armAMmass(2,1) armAMmass(2,2) armAMmass(2,3) -Mb*(yimp - y1) Mb*(ximp - x1)
           armAMmass(3,1) armAMmass(3,2) armAMmass(3,3) -Mb*(yimp - y2) Mb*(ximp - x2)];


eqns = zeros(5,5);
eqns(1, :) = totalAMmass(1, :);
eqns(2, :) = totalAMmass(2, :);
eqns(3, :) = totalAMmass(3, :);
eqns(4, :) = [ l2*cos(q1 + q2 - thetaPerp) + limp*cos(q1 + q2 + q3 - thetaPerp) + l1*cos(q1 - thetaPerp), l2*cos(q1 + q2 - thetaPerp) + limp*cos(q1 + q2 + q3 - thetaPerp), limp*cos(q1 + q2 + q3 - thetaPerp),  cos(thetaPerp), sin(thetaPerp)];
eqns(5, :) = -coeffRestitution*[ l2*sin(q1 + q2 - thetaPerp) + limp*sin(q1 + q2 + q3 - thetaPerp) + l1*sin(q1 - thetaPerp), l2*sin(q1 + q2 - thetaPerp) + limp*sin(q1 + q2 + q3 - thetaPerp), limp*sin(q1 + q2 + q3 - thetaPerp), -sin(thetaPerp), cos(thetaPerp)];
LHS = eqns*X(6:10);

eqns(5, :) = [ l2*sin(q1 + q2 - thetaPerp) + limp*sin(q1 + q2 + q3 - thetaPerp) + l1*sin(q1 - thetaPerp), l2*sin(q1 + q2 - thetaPerp) + limp*sin(q1 + q2 + q3 - thetaPerp), limp*sin(q1 + q2 + q3 - thetaPerp), -sin(thetaPerp), cos(thetaPerp)];
RHSmatrix = eqns;


dxpost = RHSmatrix\LHS;

Xpost = [X(1:5);dxpost];

end





