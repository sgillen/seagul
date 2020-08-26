%% Function to impose constraints on motion
function flag = constraint(X)

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

if yb < 0
    %condition = "Ball below Link 2";
    flag = 4;
elseif y3 < 0
    %condition = "Link1 below ground";
    flag = 3;
elseif y2 < 0
    %condition = "Link2 below ground";
    flag = 2;
elseif y1 < 0
    %condition = "Link1 below ground";
    flag = 1;
else
    flag = 0;
    %condition = "No Failure";
end