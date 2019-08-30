% This is clearly a hack, it's only here to debug the python code I'm working on 

function [dX,u] = acrobot_eom(t,X,obj)

            %we can remove these, but I think it makes the code more
            %readable and the performance hit is neglible (if not optimized
            %away entirely) 
            u = 0;
            
            th1 = X(1);
            th2 = X(2);
            dth1 = X(3);
            dth2 = X(4);
            
            %Inertia matrix (M) and conservative torque terms (C)
            %may be able to save some time by not computing non theta dependent values
            %everyime, but probably not worthwhile.  
            % lc2*m2*sin(th1 - th2)*dth2^2*l1 + d2th1*m2*l1^2 + d2th2*lc2*m2*cos(th1 - th2)*l1 + d2th1*m1*lc1^2 - g*m1*cos(th1)*lc1 + J1*d2th1
            %eq2 = J2*d2th2 + d2th2*lc2^2*m2 - g*lc2*m2*cos(th2) + d2th1*l1*lc2*m2*cos(th1 - th2) - dth1^2*l1*lc2*m2*sin(th1 - th2)

            M11 = obj.J1 + obj.m2*obj.L1^2 + obj.m1*obj.m1*obj.L1c^2;
            M12 = obj.L2c*obj.m2*cos(th1 - th2)*obj.L1 ;
            M21 = obj.L1*obj.L2c*obj.m2*cos(th1 - th2);
            M22 = obj.J2 + obj.L2c^2*obj.m2; 
            
            %C1 = obj.g*obj.m1*cos(th1)*obj.L1c + obj.L2c*obj.m2*sin(th1 - th2)*dth2^2*obj.L1 ;
            %C2 = obj.g*obj.L2c*obj.m2*cos(th2) +  dth1^2*obj.L1*obj.L2c*obj.m2*sin(th1 - th2);
            
            C1 = obj.g*obj.m1*cos(th1)*obj.L1c + obj.L2c*obj.m2*sin(th1 - th2)*dth2^2*obj.L1 ;
            C2 = obj.g*obj.L2c*obj.m2*cos(th1 + th2) +  dth1^2*obj.L1*obj.L2c*obj.m2*sin(th1 - th2);
            
            
            M = [M11, M12; M21, M22];
            C = [C1; C2];
            
            % M*d2th + C = Xi, where Xi are the non-conservative torques, i.e.,
            % Xi = [0; tau2; tau3].
            % Let u = [tau2; tau3], and Xi = [0 0; 1 0; 0 1]*u =
            % So, dX = AX + Bu formulation yields B = [zeros(3,2); inv(M)*[0 0;1 0;0 1]
           
            % Combine states to define parameters to be directly controlled:
            % TODO, unwrap our angles
            
            umat = [0; 1]; % Which EOMs does u affect?
            
            b = .01;
            B = [-b*X(3); -b*X(4)]; %Let's throw some damping in
            d2th = M \ (-C + umat*u + B);
%            if(rcond(M) < 1e-15)
%                d2th
%            end
            dth = X(3:4); % velocity states, in order to match positions...
            dX = [dth; d2th];
            