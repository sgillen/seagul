X0 = [-pi/2+.1;0;0;0];
[tl,yl] = ode45(@(t,y)acrobot_lstm(t,y,trainedNet),[0 20],X0);
figure(1);
%M = acrobot_animate(t,y);
tau = 0*tl;
for n=1:length(tl)
    [dx,tau(n)] = acrobot_lstm(tl(n),yl(n,:)',trainedNet);
end
figure(2); subplot(211); plot(tl,tau);
M = acrobot_animate(tl,yl);
