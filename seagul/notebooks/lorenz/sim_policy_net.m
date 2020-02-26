
% This comes from the code used to transfer weights to the simulink bots 
% In that setting you can't iterate through cell arrays, and you can't
% call external functions, that's why this code is gross.

dtype = 'float32';
hsize = 32;
in_size = 4;
out_size = 2;
num_weights = (hsize^2) + hsize*9;
load policy_map_r_euler
%load policy_map_r

net_dir = 'euler_net/';
weight_names = {'wght0.dat', 'wght1.dat', 'wght2.dat'};
bias_names   = {'bias0.dat', 'bias1.dat', 'bias2.dat'};

weights = {single(rand(in_size,hsize)/sqrt(num_weights)), single(rand(hsize,hsize)/sqrt(num_weights)), single(rand(hsize,out_size*2)/sqrt(num_weights))};
biases = {single(rand(hsize,in_size)/sqrt(num_weights)), single(rand(hsize,1)/sqrt(num_weights)), single(rand(out_size*2,1)/sqrt(num_weights))};

f = fopen(strcat(net_dir, weight_names{1}), 'r'); weights{1} = fread(f, [hsize,in_size], dtype); fclose(f);
f = fopen(strcat(net_dir, weight_names{2}), 'r'); weights{2} = fread(f, [hsize, hsize], dtype); fclose(f);
f = fopen(strcat(net_dir, weight_names{3}), 'r'); weights{3} = fread(f, [out_size*2,hsize], dtype); fclose(f);

f = fopen(strcat(net_dir, bias_names{1}), 'r'); biases{1} = fread(f, [1, hsize], dtype); fclose(f);
f = fopen(strcat(net_dir, bias_names{2}), 'r'); biases{2} = fread(f, [1, hsize], dtype); fclose(f);
f = fopen(strcat(net_dir, bias_names{3}), 'r'); biases{3} = fread(f, [1,out_size*2], dtype); fclose(f);


dxf = x_means; dyf = y_means;
minval = -25; maxval = 25; % clip "velocity" (actions), to these lims

[X,Y,Z] = meshgrid(x_eval,y_eval,z_eval);
x_lim = [min(x_eval), max(x_eval)];
y_lim = [min(y_eval), max(y_eval)];
z_lim = [min(z_eval), max(z_eval)];

Rstart = 10.0;
%X0 = [2.5000    0.4651    0.1250    Rstart]';
X0 = [1 0 .3 Rstart]';  % reaches the value below (close to a limit cycle)
%X0 = [-22.628522872924805   6.247414588928223  -0.975820839405060 -10.000000000000000];

dt = .01;
nact = 10; % nact dt's occur before next action update...

ttot = []; xtot = [];
addt = 0;
for nswap = 1:500
    [nswap, 20]
    
    tout = addt+[0:dt:100]';
    xout = zeros(length(tout),4);
    xout(1,:) = X0';
    
    if X0(4)<0
        ri = 1;
    else
        ri = 2; % index for R
    end
    n=1;
    bGo = true; % not yet time to swap sign of R
    a=1; % not within goal region
    while (n<length(tout)) && (bGo==true)
        n = n+1;
        
        x = max(x_lim(1),min(x_lim(2),xout(n-1,1)));
        y = max(y_lim(1),min(y_lim(2),xout(n-1,2)));
        z = max(z_lim(1),min(z_lim(2),xout(n-1,3)));
        
        dz = x;
        dr = 0;
        if mod(n-1,10) == 1
            
            if ri == 1
                r = -10;
            else
                r = 10;
            end
                
            xyzr = [x,y,z,r];
            % for sure can throw this in a function
            x1 = tanh(xyzr*weights{1}' + biases{1});
            x2 = tanh(x1*weights{2}' + biases{2});
            out = x2*weights{3}' + biases{3};
            means = out(1:2); std = out(3:4);
            dx = means(1); dy = means(2);
            
            %xyzr
            %diffx = interp3(X,Y,Z,dxf(:,:,:,ri),x,y,z,'linear') - dx
            %diffy = interp3(X,Y,Z,dyf(:,:,:,ri),x,y,z,'linear') - dy

        end
        
        dz = max(minval,min(maxval,dz));
        dy = max(minval,min(maxval,dy));
        dx = max(minval,min(maxval,dx));
        xout(n,:) = xout(n-1,:) + dt * [dx, dy, dz, dr];
        if a~=0
            [a,b,c] = ode_event_Rvec(tout(n),xout(n,:)');
            % Once it EVER detects a=0, keep this signal for the
            % full nact time steps
        end
        if (a==0) && (mod(n,nact) == 1)
            bGo = false;
            tout = tout(1:n);
            xout = xout(1:n,:);
            thit = tout(n);
        end
        
    end
    if a~=0
        fprintf('Goal was never attained...\n')
        thit = tout(n);
    end
    
    X0 = xout(end,:)'
    if a==0
        X0(4) = -X0(4);
    end
    xtot = [xtot; xout(1:end-1,:)];
    ttot = [ttot; tout(1:end-1)];
    addt = thit;
    figure(11); plot(ttot,xtot); drawnow
end
xtot = [xtot; xout(end,:)];
ttot = [ttot; tout(end)];



figure(21); clf;
plot3(xtot(:,1),xtot(:,2),xtot(:,3),'.-'); hold on
plot3(xtot(1,1),xtot(1,2),xtot(1,3),'rp')

% draw goal region boxes:
xb = [2 12]; zb = [3 13];
xsurf = xb([1 2 2 1 1]);
ysurf = -10+20*[-1 -1 1 1 -1];
zsurf = zb([1 2 2 1 1]);
for si = [1 -1]
    patch(si*xsurf,ysurf,zb(1)*si+0*xsurf,'c','facealpha',.5)
    patch(si*xsurf,ysurf,zb(2)*si+0*xsurf,'c','facealpha',.5)
    patch(0*zsurf+xb(1)*si,ysurf,si*zsurf,'c','facealpha',.5)
    patch(0*zsurf+xb(2)*si,ysurf,si*zsurf,'c','facealpha',.5)
end

axis vis3d; grid on
%figure(22); clf
%plot(tout,xout)
%close
% if 1
%     ttot = tout; xtot = xout;
%     for nswap = 1:100
%         nswap
%         X0 = xout(end,:);
%         X0(end) = -X0(end); % swap value of R...
%         [tout,xout] = ode45(@get_policy_sean,[0 100],X0,S);
%         tout(end)
%         ttot = [ttot(1:end-1); ttot(end)+tout];
%         xtot = [xtot(1:end-1,:); xout];
%     end
% end
%
% figure(33); clf
% plot3(xtot(:,1),xtot(:,2),xtot(:,3)); hold on
% patch(xsurf,ysurf,zb(1)+0*xsurf,'c','facealpha',.5)
% patch(xsurf,ysurf,zb(2)+0*xsurf,'c','facealpha',.5)
% zsurf = zb([1 2 2 1 1]);
% patch(0*zsurf+xb(1),ysurf,zsurf,'c','facealpha',.5)
% patch(0*zsurf+xb(2),ysurf,zsurf,'c','facealpha',.5)
%
% patch(-xsurf,ysurf,-zb(1)+0*xsurf,'c','facealpha',.5)
% patch(-xsurf,ysurf,-zb(2)+0*xsurf,'c','facealpha',.5)
% zsurf = zb([1 2 2 1 1]);
% patch(0*zsurf-xb(1),ysurf,-zsurf,'c','facealpha',.5)
% patch(0*zsurf-xb(2),ysurf,-zsurf,'c','facealpha',.5)
% grid on
% axis vis3d
%
%
%
%
