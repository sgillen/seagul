function M = acrobot_animate(t_ode,X_ode)

M=[];
dt = .05;
tlist = min(t_ode):dt:max(t_ode);
Xlist = zeros(4,length(tlist));
for i=1:4
    Xlist(i,:) = interp1(t_ode,X_ode(:,i),tlist);
end
clear M
for n=1:length(tlist)
    acrobot_draw(Xlist(:,n)); drawnow; pause(dt*.05);
    M(n) = getframe(gca);
end
