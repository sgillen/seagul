%% Set parameters
clc
clear all
close all
Ts = 0.002;
x0=[0.1,0,0,0].'; % [x1 x2 x1_dot x2_dot]


m1 = 0.257; % m1=0.2;%kg
m2= 0.127; %kg
L1 = 0.216 ;%m
L2 = 0.337;%m
l1 = 0.0619; %L1/2 ; % location of the masse
l2 = 0.1556; %L2/2 ;
g=9.81; %m/sec^2
J1 =  m1*L1^2/12; %kg*m^2
J2 =  m2*L2^2/12; %kg*m^2
b1 = 0.0024; % Damping
b2 = 0.0024; % Damping
J1_ = J1 + m1*l1^2;
J2_ = J2 + m2*l2^2;
J0_ = J1_ + m2*L1^2 

%  current Mechanik und max
L = 0.18e-3 ; %H
R = 2.75 ; % Ohm
Km = 7.86e-3; % Nm/A
V_max = 5;
I_max = 0.5;
nu = 0.9; % wirkungsgrad
N = 70; %übersetzung
Torque_max = Km*I_max*nu*N; %0.2476 Nm

%% Störgrößenaufschaltung State estimator - Beobachter around x = [0;0;0;0]
% 
% b1 = 0.0024; % Damping
% b2 = 0.0024; % Damping

A = [ 0,0,1,0 ; 0,0,0,1; ...
         0, (L1*g*l2^2*m2^2)/(- L1^2*l2^2*m2^2 + J0_*J2_),  ...
         -(J2_*b1)/(- L1^2*l2^2*m2^2 + J0_*J2_), ...
         (L1*b2*l2*m2)/(- L1^2*l2^2*m2^2 + J0_*J2_); ...
         0,   -(J0_*g*l2*m2)/(- L1^2*l2^2*m2^2 + J0_*J2_), ...
         (L1*b1*l2*m2)/(- L1^2*l2^2*m2^2 + J0_*J2_),   ...
         -(J0_*b2)/(- L1^2*l2^2*m2^2 + J0_*J2_)];
B =[0; 0; J2_/(- L1^2*l2^2*m2^2 + J0_*J2_) ; ...
         -(L1*l2*m2)/(- L1^2*l2^2*m2^2 + J0_*J2_) ];
C = [1 0 0 0; 0 1 0 0];
D = 0;
Bz = [0 0; 0 0; 1 0; 0 1];
A_aug = [A Bz; zeros(2,6)];
C_aug = [C zeros(2,2)];
B_aug = [B ;0;0];
w_estimator = -60; % mindestens 5 mal schneller als streckenpole, abtastfrequenz fragen

 % checken ob Systeme Steuerbar und Beobachtbar (Kriterium von Kalman)
By = [C_aug ; C_aug*A_aug; C_aug*A_aug^2; ; C_aug*A_aug^3; C_aug*A_aug^4; C_aug*A_aug^5] ; %rank checken
[V1,D1,W1] = eig(A_aug);
AJ1  = D1; % A matrix in Jordan Form: eigenwerte auf Diagonalen
TJ1  = V1; % rechtsseite Eigenvektoren :  Transformationsmatrix 
TJ1(5,5)=TJ1(1,5); TJ1(6,6)=TJ1(1,6); 
TJ1_ = inv(TJ1); % linkseitige Eigenvektoren :  inverse Transformationsmatrix 
LJ1 = [AJ1(1,1)-w_estimator 0; 0 AJ1(2,2)-w_estimator];
L1 = TJ1(:,1:2)*LJ1*inv( C_aug*TJ1(:,1:2) );
% Pol 3 und 4 verschieben
A1 = (A_aug-L1*C_aug);
[V1,D1,W1] = eig(A1);
D1
AJ1  = D1; % A matrix in Jordan Form: eigenwerte auf Diagonalen
TJ1  = V1; % rechtsseite Eigenvektoren :  Transformationsmatrix 
TJ1_ = inv(V1); % linkseitige Eigenvektoren :  inverse Transformationsmatrix 

LJ1 = [AJ1(3,3)-w_estimator 0; 0 AJ1(4,4)-w_estimator];
L2 = TJ1(:,3:4)*LJ1*inv( C_aug*TJ1(:,3:4) );
% Pol 5 und 6 verschieben
A2 = (A_aug-(L1+L2)*C_aug);
[V1,D1,W1] = eig(A2);
AJ1  = D1; % A matrix in Jordan Form: eigenwerte auf Diagonalen
TJ1  = V1; % rechtsseite Eigenvektoren :  Transformationsmatrix 
TJ1_ = inv(V1); % linkseitige Eigenvektoren :  inverse Transformationsmatrix 

LJ1 = [AJ1(5,5)-w_estimator 0; 0 AJ1(6,6)-w_estimator];
L3 = TJ1(:,5:6)*LJ1*inv( C_aug*TJ1(:,5:6) );

L_aug = real (L1+L2+L3);
eig(A_aug-L_aug*C_aug)

%% Controller 4 pole verschieben 
% Trafo in RNF
Su =inv([B A*B A^2*B A^3*B]);
Tr = [Su(end,:) ; Su(end,:)*A; Su(end,:)*A^2;Su(end,:)*A^3];
Ar = Tr*A*inv(Tr)
Br = Tr*B 

%
a4 = 1; a0 = 0; a1 =29.8113 ; a2 =98.6130   ; a3 =1.8740; % eq x =0
% a4 = 1; a0 = 0; a1 = -7.6564; a2 =-15.2116 ; a3 =1.9496;% eq x =pi
w0  = 8;
% binominal filter s^4
p0 = w0^4 ; p1 = 4*w0^3; p2 = 6*w0^2; p3 = 4*w0 ; p4 =1;
% butterwoth filter s^4
% p0 = w0^4 ; p1 = sqrt(4+2*sqrt(2))*w0^3; p2 = (2+sqrt(2))*w0^2; p3 = sqrt(4+2*sqrt(2))*w0 ; p4 =1;

r = [p0 ; p1 - a1; p2-a2; p3-a3];
eig(Ar)
eig(Ar-Br*r')


% Störgrößenaufschaltung
Bz = [0 0; 0 0; 1 0; 0 1];
Bzr = Tr*Bz; 
Cz1 = [1 0 0 0]; 
Cr1 = Cz1* inv(Tr);
Qz =  - inv(Cr1*inv(Ar-Br*r')*Br)*Cr1*inv(Ar-Br*r')*Bzr;


%% REST ALT
%%% Calculation jacobian
% syms x1 x2 x1_dot x2_dot torque m1 m2 L1 L2 l1 l2 g J1 J2 b1 b2 J1_ J2_ J0_
% 
% a1 = -J2_*b1*x1_dot;
% a2 = m2*L1*l2*cos(x2)*b2*x2_dot;
% a3 = -J2_^2*sin(2*x2)*x1_dot*x2_dot;
% a4 = -(1/2)*J2_*m2*L1*l2*cos(x2)*sin(2*x2)*x1_dot^2;
% a5 = J2_*m2*L1*l2*sin(x2)*x2_dot^2;
% a6 = J2_ * torque;
% % a7 torque 2
% a8 = 1/2*m2^2*l2^2*L1*sin(2*x2)*g;
% nenner= J0_*J2_+J2_^2*sin(x2)^2-m2^2*L1^2*l2^2*cos(x2)^2;
% x1_dot_dot = (a1+a2+a3+a4+a5+a6+a8)/nenner;
% 
% 
% b1 = m2*L1*l2*cos(x2)*b1*x1_dot;
% b2 = -b2*(J0_+J2_*sin(x2)^2)*x2_dot;
% b3 = m2*L1*l2*J2_*cos(x2)*sin(2*x2)*x1_dot*x2_dot;
% b4 = -1/2*sin(2*x2)*(J0_*J2_+J2_^2*sin(x2)^2)*x1_dot^2;
% b5 = -1/2*m2^2*L1^2*l2^2*sin(2*x2)*x2_dot^2;
% b6 = -m2*L1*l2*cos(x2)*torque;
% % b7 torque 2
% b8 = -m2*l2*sin(x2)*(J0_+J2_*sin(x2)^2)*g;
% x2_dot_dot = (b1+b2+b3+b4+b5+b6+b8)/nenner;
% 
% x = [x1;x2; x1_dot; x2_dot] ;
% u = [torque] ;
% f = [x1_dot_dot ; x2_dot_dot] ;
% 
% A_jacobian = jacobian(f,x);
% B_jacobian = jacobian(f,u);
% 
% %% linearisation around x = [0;0;0;0]
% 
% A_0000 = subs(A_jacobian,[u ; x] ,[0;0;0;0;0]) ;
% B_0000 = subs(B_jacobian,[u ; x] ,[0;0;0;0;0]) ;
% 
% A_0000 = [ 0,0,1,0 ; 0,0,0,1; ...
%          0, (L1*g*l2^2*m2^2)/(- L1^2*l2^2*m2^2 + J0_*J2_),  ...
%          -(J2_*b1)/(- L1^2*l2^2*m2^2 + J0_*J2_), ...
%          (L1*b2*l2*m2)/(- L1^2*l2^2*m2^2 + J0_*J2_); ...
%          0,   -(J0_*g*l2*m2)/(- L1^2*l2^2*m2^2 + J0_*J2_), ...
%          (L1*b1*l2*m2)/(- L1^2*l2^2*m2^2 + J0_*J2_),   ...
%          -(J0_*b2)/(- L1^2*l2^2*m2^2 + J0_*J2_)];
% B_0000 =[0; 0; J2_/(- L1^2*l2^2*m2^2 + J0_*J2_) ; ...
%          -(L1*l2*m2)/(- L1^2*l2^2*m2^2 + J0_*J2_) ];
% 
% %% linearisation around x = [0;pi;0;0]
% 
% A_0000 = subs(A_jacobian,[x ; u] ,[0;pi;0;0;0]) ;
% B_0000 = subs(B_jacobian,[x ; u] ,[0;pi;0;0;0]) ;
% 
% A_0000 = [ 0,0,1,0 ; 0,0,0,1; ...
%          0, (L1*g*l2^2*m2^2)/(- L1^2*l2^2*m2^2 + J0_*J2_),  ...
%          -(J2_*b1)/(- L1^2*l2^2*m2^2 + J0_*J2_), ...
%          -(L1*b2*l2*m2)/(- L1^2*l2^2*m2^2 + J0_*J2_); ...
%          0,   (J0_*g*l2*m2)/(- L1^2*l2^2*m2^2 + J0_*J2_), ...
%          -(L1*b1*l2*m2)/(- L1^2*l2^2*m2^2 + J0_*J2_),   ...
%          -(J0_*b2)/(- L1^2*l2^2*m2^2 + J0_*J2_)];
% B_0000 =[0; 0; J2_/(- L1^2*l2^2*m2^2 + J0_*J2_) ; ...
%          (L1*l2*m2)/(- L1^2*l2^2*m2^2 + J0_*J2_) ];
% 
% %% State estimator - Beobachter around x = [0;pi;0;0]
% 
% b1 = 0.01; % Damping
% b2 = 0.02; % Damping
% 
% A = [ 0,0,1,0 ; 0,0,0,1; ...
%          0, (L1*g*l2^2*m2^2)/(- L1^2*l2^2*m2^2 + J0_*J2_),  ...
%          -(J2_*b1)/(- L1^2*l2^2*m2^2 + J0_*J2_), ...
%          -(L1*b2*l2*m2)/(- L1^2*l2^2*m2^2 + J0_*J2_); ...
%          0,   (J0_*g*l2*m2)/(- L1^2*l2^2*m2^2 + J0_*J2_), ...
%          -(L1*b1*l2*m2)/(- L1^2*l2^2*m2^2 + J0_*J2_),   ...
%          -(J0_*b2)/(- L1^2*l2^2*m2^2 + J0_*J2_)];
% B =[0; 0; J2_/(- L1^2*l2^2*m2^2 + J0_*J2_) ; ...
%          (L1*l2*m2)/(- L1^2*l2^2*m2^2 + J0_*J2_) ];
% C = [1 0 0 0; 0 1 0 0];
%      
% w_estimator = 100; % mindestens 5 mal schneller als streckenpole, abtastfrequenz fragen
% 
%  % checken ob Systeme Steuerbar und Beobachtbar (Kriterium von Kalman)
% By = [C ; C*A; C*A^2; ; C*A^3] ; %rank checken
% [V1,D1,W1] = eig(A);
% AJ1  = D1; % A matrix in Jordan Form: eigenwerte auf Diagonalen
% TJ1  = V1; % rechtsseite Eigenvektoren :  Transformationsmatrix 
% TJ1_ = inv(V1); % linkseitige Eigenvektoren :  inverse Transformationsmatrix 
% 
% LJ1 = [AJ1(1,1)-w_estimator 0; 0 AJ1(2,2)-w_estimator];
% L1 = TJ1(:,1:2)*LJ1*inv( C*TJ1(:,1:2) );
% % Pol 3 und 4 verschieben
% A1 = (A-L1*C);
% [V1,D1,W1] = eig(A1);
% D1
% AJ1  = D1; % A matrix in Jordan Form: eigenwerte auf Diagonalen
% TJ1  = V1; % rechtsseite Eigenvektoren :  Transformationsmatrix 
% TJ1_ = inv(V1); % linkseitige Eigenvektoren :  inverse Transformationsmatrix 
% 
% LJ1 = [AJ1(3,3)-w_estimator 0; 0 AJ1(4,4)-w_estimator];
% L2 = TJ1(:,3:4)*LJ1*inv( C*TJ1(:,3:4) );
% eig(A-(L1+L2)*C)
% 
% L = real (L1 + L2);
%          
% %% State estimator - Beobachter around x = [0;0;0;0]
% 
% b1 = 0.01; % Damping
% b2 = 0.02; % Damping
% 
% A = [ 0,0,1,0 ; 0,0,0,1; ...
%          0, (L1*g*l2^2*m2^2)/(- L1^2*l2^2*m2^2 + J0_*J2_),  ...
%          -(J2_*b1)/(- L1^2*l2^2*m2^2 + J0_*J2_), ...
%          (L1*b2*l2*m2)/(- L1^2*l2^2*m2^2 + J0_*J2_); ...
%          0,   -(J0_*g*l2*m2)/(- L1^2*l2^2*m2^2 + J0_*J2_), ...
%          (L1*b1*l2*m2)/(- L1^2*l2^2*m2^2 + J0_*J2_),   ...
%          -(J0_*b2)/(- L1^2*l2^2*m2^2 + J0_*J2_)];
% B =[0; 0; J2_/(- L1^2*l2^2*m2^2 + J0_*J2_) ; ...
%          -(L1*l2*m2)/(- L1^2*l2^2*m2^2 + J0_*J2_) ];
% C = [1 0 0 0; 0 1 0 0];
% D = 0;
% sysc = ss(A,B,C,D);
% sysd = c2d(sysc,Ts);
% Ad = sysd.A; Bd = sysd.B ; Cd = sysd.C; 
% w_estimator = -100; % mindestens 5 mal schneller als streckenpole, abtastfrequenz fragen
% 
%  % checken ob Systeme Steuerbar und Beobachtbar (Kriterium von Kalman)
% By = [C ; C*A; C*A^2; ; C*A^3] ; %rank checken
% [V1,D1,W1] = eig(A);
% AJ1  = D1; % A matrix in Jordan Form: eigenwerte auf Diagonalen
% TJ1  = V1; % rechtsseite Eigenvektoren :  Transformationsmatrix 
% TJ1_ = inv(V1); % linkseitige Eigenvektoren :  inverse Transformationsmatrix 
% 
% LJ1 = [AJ1(1,1)-w_estimator 0; 0 AJ1(2,2)-w_estimator];
% L1 = TJ1(:,1:2)*LJ1*inv( C*TJ1(:,1:2) );
% % Pol 3 und 4 verschieben
% A1 = (A-L1*C);
% [V1,D1,W1] = eig(A1);
% D1
% AJ1  = D1; % A matrix in Jordan Form: eigenwerte auf Diagonalen
% TJ1  = V1; % rechtsseite Eigenvektoren :  Transformationsmatrix 
% TJ1_ = inv(V1); % linkseitige Eigenvektoren :  inverse Transformationsmatrix 
% 
% LJ1 = [AJ1(3,3)-w_estimator 0; 0 AJ1(4,4)-w_estimator];
% L2 = TJ1(:,3:4)*LJ1*inv( C*TJ1(:,3:4) );
% eig(A-(L1+L2)*C)
% 
% L = real (L1 + L2);
%      
% 
%      
% 
