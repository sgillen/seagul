%Haptics Lab 2
%Neeli Tummala and Paige Sullivan

Hapticsdata = load('Part2.csv');

figure(1)
plot(t(2595:end),px(2595:end))
xlabel('Time (sec)')
ylabel('X Position (m)')
title('X Position v Time')

figure(2)
plot(t(2595:end),py(2595:end))
xlabel('Time (sec)')
ylabel('Y Position (m)')
title('Y Position v Time')

figure(3)
plot(t(2595:end),pz(2595:end))
xlabel('Time (sec)')
ylabel('Z Position (m)')
title('Z Position v Time')

figure(4)
plot(t(2595:end),fx(2595:end))
xlabel('Time (sec)')
ylabel('X Force (N)')
title('X Force v Time')

figure(5)
plot(t(2595:end),fy(2595:end))
xlabel('Time (sec)')
ylabel('Y Force (N)')
title('Y Force v Time')

figure(6)
plot(t(2595:end),fz(2595:end))
xlabel('Time (sec)')
ylabel('Z Force (N)')
title('Z Force v Time')


T = 0.337;

K = 500; 

%% Annotate x position
figure(1)
plot(t(2595:end),px(2595:end))
xlabel('Time (sec)')
ylabel('X Position (m)')
title('X Position v Time')
hold on
tlen = size(t);
time = linspace(0,t(end)-t(2595), tlen(1)-2595+1);
plot(t(2595:end), 0.05*exp(-time*omega*alpha)+0.003,'r--');
plot(t(2595:end), -0.05*exp(-time*omega*alpha)+0.003,'r--');

%% wall

figure(1)
plot(cursorpos.t,cursorpos.px)
xlabel('Time (sec)')
ylabel('X Position (m)')
title('Virtual Wall')

figure(2)
plot(cursorpos.t,cursorpos.py)
xlabel('Time (sec)')
ylabel('Y Position (m)')
title('Virtual Wall')

figure(3)
plot(cursorpos.t,cursorpos.pz)
xlabel('Time (sec)')
ylabel('Z Position (m)')
title('Virtual Wall')

figure(4)
plot(cursorpos.t,cursorpos.px,cursorpos.t,cursorpos.py,cursorpos.t,cursorpos.pz)
xlabel('Time (sec)')
ylabel('X Position (m)')
title('Virtual Wall')
legend('x pos','y pos','z pos')

figure(5)
plot(cursorpos.t,cursorpos.fx)
xlabel('Time (sec)')
ylabel('X Force (N)')
title('Virtual Wall')

figure(6)
plot(cursorpos.t,cursorpos.fy)
xlabel('Time (sec)')
ylabel('Y Force (N)')
title('Virtual Wall')

figure(7)
plot(cursorpos.t,cursorpos.fz)
xlabel('Time (sec)')
ylabel('Z Force (N)')
title('Virtual Wall')

%% damping

figure(1)
plot(damperpos.t,damperpos.px)
xlabel('Time (sec)')
ylabel('X Position (m)')
title('Virtual Wall, with damping')

figure(2)
plot(damperpos.t,damperpos.py)
xlabel('Time (sec)')
ylabel('Y Position (m)')
title('Virtual Wall, with damping')

figure(3)
plot(damperpos.t,damperpos.pz)
xlabel('Time (sec)')
ylabel('Z Position (m)')
title('Virtual Wall, with damping')

figure(4)
plot(damperpos.t,damperpos.px,damperpos.t,damperpos.py,damperpos.t,damperpos.pz)
xlabel('Time (sec)')
ylabel('X Position (m)')
title('Virtual Wall, with damping')
legend('X position','Y position','Z position')

figure(5)
plot(damperpos.t,damperpos.fx)
xlabel('Time (sec)')
ylabel('X force (m)')
title('Virtual Wall, with damping')

figure(6)
plot(damperpos.t,damperpos.fy)
xlabel('Time (sec)')
ylabel('Y force (m)')
title('Virtual Wall, with damping')

figure(7)
plot(damperpos.t,damperpos.fz)
xlabel('Time (sec)')
ylabel('Z force (m)')
title('Virtual Wall, with damping')
