walker = CGWalker

walker.L1 = 1;
walker.L2 = 1;
walker.L1c = .5;
walker.L2c = .5;
walker.m1 = 1;
walker.m2 = 1;
walker.J1 = 1;
walker.J2 = 1;

[t,x] = ode45( @(t,y)acrobot_eom(t, y, walker), [0,20], [-pi/2,0,0,0] )
walker.animateStep(t,x',[0,0]')