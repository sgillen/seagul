function [val, isterm, dir] = ode_event_R(t,Xin)

% vectorized version... Xin is 4xN

% terminations occur where val=0
% val maps to a distance from the relevant "reward box", roughly
% (when in the box, val=0 exactly...)

x = Xin(1,:); y = Xin(2,:); z = Xin(3,:); r = Xin(4,:);


xb = [2 12]; zb = [3 13]; % true boxes
%xb = [2 20]; zb = [3 20]; % testing..
mx = mean(xb); mz = mean(zb);
sx = xb(2)-mx; sz = zb(2)-mz;


%if r > 0
dx = x-mx;
dx = max(0,abs(dx) - sx);
dz = z-mz;
dz = max(0,abs(dz) - sz);
val = (dx + dz).*(r>0); % zero iff satisfied; otw a "distance" (approx)
%isterm = true;
%dir = -1; % decreasing
%else
%keyboard
x = -x; z = -z;
dx = x-mx;
dx = max(0,abs(dx) - sx);
dz = z-mz;
dz = max(0,abs(dz) - sz);
val = val + (dx + dz).*(r<0); % zero iff satisfied; otw a "distance" (approx)

isterm = 1 + 0*x;
dir = -1 + 0*x; % decreasing
%end

% def reward_fn(s):
%    if s[3] > 0:
%        if 12 > s[0] > 2 and 13 > s[2] > 3:
%            reward = 5.0
%            s[3] = -10
%        else:
%            reward = 0.0
%
%    elif s[3] < 0:
%        if -12 < s[0] < -2 and -13 < s[2] < -3:
%            reward = 5.0
%            s[3] = 10
%        else:
%            reward = 0.0
%
%    return reward, s

