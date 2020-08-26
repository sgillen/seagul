%% Function to compute the inverse kinematics of a 3-link arm

function [th1, th2rel, th3rel] = inv_kin(xee,yee, theta_ee_desired)

params;

a = theta_ee_desired; % Theta_ee will be closest to this value
b = -pi;
c = pi;
precision = 1e-2;

n = 1;
th3_try = a;

while(1)
    if ( (a+(n*precision)) <= c )
        th3_try = [th3_try; (a+(n*precision))];
    end
    if ( (a-(n*precision)) >= b )
        th3_try = [th3_try; (a-(n*precision))];
    end
    if ( ( (a+(n*precision)) > c ) && ( (a-(n*precision)) < b ) )
        break;
    end
    n = n + 1;
end

x2 = xee - l3*cos(th3_try);
y2 = yee - l3*sin(th3_try);
r2 = (x2.^2 + y2.^2).^.5;
can_reach = (r2<=(l1+l2)) .* (r2>=abs(l1-l2));
fi = find(can_reach);
th3 = th3_try(fi);
x2 = x2(fi); y2 = y2(fi);
a = l1; b = l2; c = r2(fi);
theta_0_to_2 = atan2(y2,x2);
tha = acos((b.^2+c.^2-a.^2)./(2*b.*c));
thb = acos((a.^2+c.^2-b.^2)./(2*a.*c));
thc = acos((a.^2-c.^2+b.^2)./(2*a.*b));

% Two solutions exist: +tha and -tha
th1a = theta_0_to_2 + thb;
th1b = theta_0_to_2 - thb;
th2a = th1a - pi + thc;
th2b = th1b + pi - thc;
th3a = th3;
th3b = th3;

th1 = [th1a, th1b];
th2 = [th2a, th2b];
th3 = [th3a, th3b];
th3rel = mod((th3-th2)+pi,2*pi)-pi;
th2rel = mod((th2-th1)+pi,2*pi)-pi;
th1 = mod(th1+pi,2*pi)-pi;

th1 = th1';
th2rel = th2rel';
th3rel = th3rel';

