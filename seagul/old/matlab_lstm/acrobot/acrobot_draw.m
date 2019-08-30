function acrobot_draw(X)

q1 = X(1); q2 = X(2); dq1 = X(3); dq2 = X(4);

%feel like these should be global
L1 = 1; L2 = 1; Lc1 = .5; Lc2 = .5;

th = .5*.1; % half-THickness of arm
c1 = [0 0 1];  % Color for link 1
c2 = [1 0 0];  % Color for link 2

avals = pi*[0:.05:1];
x1 = [0 L1 L1+th*cos(avals-pi/2) L1 0 th*cos(avals+pi/2)];
y1 = [-th -th th*sin(avals-pi/2) th th th*sin(avals+pi/2)];
r1 = (x1.^2 + y1.^2).^.5;
a1 = atan2(y1,x1);
x1draw = r1.*cos(a1+q1);  % x pts to plot, for Link 1
y1draw = r1.*sin(a1+q1);  % y pts to plot, for Link 1
x1end = L1*cos(q1);  % "elbow" at end of Link 1, x
y1end = L1*sin(q1);  % "elbow" at end of Link 1, x

x2 = [0 L2 L2+th*cos(avals-pi/2) L2 0 th*cos(avals+pi/2)];
y2 = [-th -th th*sin(avals-pi/2) th th th*sin(avals+pi/2)];
r2 = (x2.^2 + y2.^2).^.5;
a2 = atan2(y2,x2);
x2draw = x1end+r2.*cos(a2+q1+q2);  % x pts to plot, for Link 1
y2draw = y1end+r2.*sin(a2+q1+q2);  % y pts to plot, for Link 1


% now, draw the acrobot:

figure(1); clf
p1 = patch(x1draw,y1draw,'b','FaceColor',c1); hold on
p2 = patch(x2draw,y2draw,'r','FaceColor',c2);
axis equal; axis off
W = L1+L2;
axis([-W W -W W]*1.1)




