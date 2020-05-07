
%% Function to calculate velocities post impact
function Xpost = impact(X)

% linkComX = rand(10,1)

q1 = X(1); q2 = X(2); xb = X(3); yb = X(4);
dq1 = X(5); dq2 = X(6); vbx = X(7); vby = X(8);

th1 = q1;
th2 = q1 + q2;
thetaPerp = th2 + pi/2;

params;

x1 =  -l1*sin(th1);
x1cm = -p1*sin(th1);
x2 = x1 - l2*sin(th2);
x2cm = x1 - p2*sin(th2);

y1 = l1*cos(th1);
y1cm = p1*cos(th1);
y2 = y1 + l2*cos(th2);
y2cm = y1 + p2*cos(th2);

if (x2 == x1)
    disp('link 2 in vertical position');
    Xpost = zeros(8,1,'single');
    return
end

link2Slope = (y2 - y1)/(x2 - x1);
link2Intercept = y2 - link2Slope*x2;

if link2Slope < 1e-10
    ximp = xb;
    yimp = y2;
    limp = abs(xb - x1);
else
    perpLineSlope = -1/link2Slope;
    perpLineIntercept = yb - perpLineSlope*xb;

    interceptPoint = [-link2Slope 1; -perpLineSlope 1]\[link2Intercept; perpLineIntercept];

    limp = sqrt((interceptPoint(1) - x1)^2 + (interceptPoint(2) - y1)^2);

    ximp = x1 - limp*sin(th2);
    yimp = y1 + limp*cos(th2);
end

mass = [M1,M2];
inertia = [I1,I2];
linkComX = [x1cm,x2cm];
linkComY = [y1cm,y2cm];

Mx = zeros(2,2,'single');
My = zeros(2,2,'single');
Mt = zeros(2,2,'single');

% Conserving angular momentum about hinge
xp = 0;
yp = 0;

index = [1,2];
for n=1:length(index)
    Mx(1,index(n)) = -mass(index(n))*[linkComY(index(n)) - yp];
    My(1,index(n)) = mass(index(n))*[linkComX(index(n)) - xp];
    Mt(1,index(n)) = inertia(index(n));
    
end

% Conserving angular momentum about joint 1

xp = x1;
yp = y1;

index = [2];

for n=1:length(index)
    Mx(2,index(n)) = -mass(index(n))*[linkComY(index(n)) - yp];
    My(2,index(n)) = mass(index(n))*[linkComX(index(n)) - xp];
    Mt(2,index(n)) = inertia(index(n));    
end


% Now the Jacobians

JX = [[                    -p1*cos(q1),                0]
      [ - p2*cos(q1 + q2) - l1*cos(q1), -p2*cos(q1 + q2)]];
  
JY = [[                    -p1*sin(q1),                0]
      [ - p2*sin(q1 + q2) - l1*sin(q1), -p2*sin(q1 + q2)]];

Jth = [1 0; 1 1];


armAMmass = Mx*JX + My*JY + Mt*Jth;
totalAMmass = [armAMmass(1,1) armAMmass(1,2) -Mb*yimp Mb*ximp
           armAMmass(2,1) armAMmass(2,2) -Mb*(yimp - y1) Mb*(ximp - x1)];


eqns = zeros(4,4,'single');
eqns(1, :) = totalAMmass(1, :);
eqns(2, :) = totalAMmass(2, :);
eqns(3, :) = [ limp*cos(q1 + q2 - thetaPerp) + l1*cos(q1 - thetaPerp), limp*cos(q1 + q2 - thetaPerp),  cos(thetaPerp), sin(thetaPerp)];
eqns(4, :) = -coeffRestitution*[ limp*sin(q1 + q2) + l1*sin(q1), limp*sin(q1 + q2), -sin(thetaPerp), cos(thetaPerp)];
LHS = eqns*X(5:8);

eqns(4, :) = [ limp*sin(q1 + q2 - thetaPerp) + l1*sin(q1 - thetaPerp), limp*sin(q1 + q2 - thetaPerp), -sin(thetaPerp), cos(thetaPerp)];
RHSmatrix = eqns;


dxpost = RHSmatrix\LHS;

Xpost = [X(1:4);dxpost];

end






