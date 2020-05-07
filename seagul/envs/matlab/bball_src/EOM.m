%% Code to calculate the equations of motion for 2 link arm

clear all
clc
syms q1 q2 M1 M2 l1 l2 p1 p2 I1 I2 g tau1 tau2 
syms limp e vby vbx thetaPerp

GC = {q1,q2};

th1 = q1;
th2 = q1 + q2;

x1 =  -l1*sin(th1);
x1cm = -p1*sin(th1);
x2 = x1 - l2*sin(th2);
x2cm = x1 - p2*sin(th2);
ximp = x1 - limp*sin(th2);
y1 = l1*cos(th1);
y1cm = p1*cos(th1);
y2 = y1 + l2*cos(th2);
y2cm = y1 + p2*cos(th2);
yimp = y1 + limp*cos(th2);

dq1 = fulldiff(q1,GC);
dq2 = fulldiff(q2,GC);

dx1 = fulldiff(x1,GC);
dx1cm = fulldiff(x1cm,GC);
dx2 = fulldiff(x2,GC);
dx2cm = fulldiff(x2cm,GC);
dximp = fulldiff(ximp, GC);

dy1 = fulldiff(y1,GC);
dy1cm = fulldiff(y1cm,GC);
dy2 = fulldiff(y2,GC);
dy2cm = fulldiff(y2cm,GC);
dyimp = fulldiff(yimp, GC);

dth1 = dq1
dth2 = dq2 + dq1;

d2q1 = fulldiff(dq1,GC);
d2q2 = fulldiff(dq2,GC);



for n=1:length(GC)
    dGC{n} = fulldiff(GC{n},GC);
    d2GC{n} = fulldiff(dGC{n},GC);
end

% Tstar = 0.5*Mt*(dx3cm^2 + dy3cm^2) + 0.5*Mt*(dx4cm^2 + dy4cm^2) + 0.5*Mf*(dx1cm^2 + dy1cm^2) ... 
%         + 0.5*Mf*(dx2cm^2 + dy2cm^2) + 0.5*MT*(dx5cm^2 + dy5cm^2) + 0.5*It*(dth3^2) ... 
%         + 0.5*It*(dth4^2) + 0.5*If*(dth1^2) + 0.5*If*(dth2^2) + 0.5*IT*(dth5^2) ...
%         + 0.5*Ia*(dmot1^2) + 0.5*Ia*(dmot2^2) + 0.5*Ia*(dmot3^2) + 0.5*Ia*(dmot4^2);
    

Tstar =  0.5*M1*(dx1cm^2 + dy1cm^2) + 0.5*M2*(dx2cm^2 + dy2cm^2)...
         + 0.5*I1*(dth1^2) + 0.5*I2*(dth2^2);

V = M1*g*y1cm + M2*g*y2cm;

Lag = Tstar - V;

for n=1:length(GC)
    LHS{n} = fulldiff(diff(Lag,dGC{n}),GC) - diff(Lag,GC{n});
end

for n=1:length(GC)
    LHS{n} = simplify(expand(LHS{n}));
end

for n=1:length(GC)
    for m=1:length(GC)
        D(m,n) = jacobian(LHS{m},d2GC{n});
    end
end
d2q1 = 0; d2q2 = 0;
for m=1:length(GC)
    C(m,1) = simplify(eval(LHS{m}));  
end

U = [tau1;tau2];
D_inv = inv(D);

% d2X = D_inv*(-C + U);
% for n = 1:length(GC)
%     d2X(n) = simplify(d2X(n));
%     n
% end

% Impact Calcuation

rotationMatrix = [cos(thetaPerp) sin(thetaPerp)
                  -sin(thetaPerp) cos(thetaPerp)];
              
vbperp = rotationMatrix*[vbx; vby];

vlinkperp = rotationMatrix*[dximp; dyimp];

eq3 = (vbperp(1) - vlinkperp(1))
eq4 = (vbperp(2) - vlinkperp(2))

GCimp = {dq1, dq2, vbx, vby};
for i = 1:1:length(GCimp)
    coeffeq3(1, i) = simplify(jacobian(eq3, GCimp{i}));
    coeffeq4(1, i) = simplify(jacobian(eq4, GCimp{i}));
end

[coeffeq3; coeffeq4]

for i = 1:1:length(dGC)
    JX(1, i) = simplify(jacobian(dx1cm, dGC{i}));
    JX(2, i) = simplify(jacobian(dx2cm, dGC{i}));
    JY(1, i) = simplify(jacobian(dy1cm, dGC{i}));
    JY(2, i) = simplify(jacobian(dy2cm, dGC{i}));
end
JX
JY
