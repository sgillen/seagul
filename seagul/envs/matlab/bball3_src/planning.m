%% Function to plan the desired trajectory given the initial state of the system and the desired bounce height

function [t, X_desired, ee_desired, ee_dot_desired, error_in_final_desired] = planning(X, h_b_desired, timeStep, initialBounce)


%% Model parameters:

params;
l3 = l3/2; % This is to ensure that the collision occurs at the center of the 3rd link

% Positive angles are counter-clockwise from +y axis
% The "old reference" implies the inv_kin file was written considering +x axis as reference for all angles
theta_b_collision = pi; % Absolute angle of collision of the ball with 3rd link
theta_ee_final = wrapToPi( theta_b_collision/2 );
theta_ee_final_oldReference = wrapToPi( theta_ee_final + pi/2 );

xf_ee = (X(4) - l3); yf_ee = 0.4; % Coordinates of final position of ee
h_b_relative = h_b_desired - yf_ee; % Apex height of bounce from ee

% Desired x and y components of post-impact ball velocity
xb_dot_postImpact_desired = 0; % This may change if non-vertical trajectories are planned
yb_dot_postImpact_desired = (2*g*h_b_relative)^0.5;

if(initialBounce == 1)
    expectedCollisionTime = (2*(X(5)-yf_ee-rb)/g)^0.5;
else
    expectedCollisionTime = (yb_dot_postImpact_desired/g) + (2*(h_b_relative-rb)/g)^0.5;
end

timeSpan = [0, expectedCollisionTime]; % Time span of current bounce cycle
t = [timeSpan(1):timeStep:timeSpan(2)]'; % Discretized time instants during current bounce cycle

discrete_pts = length(t); % Number of discretized time instants


%% Initialization of desired values:

q_desired = zeros(discrete_pts, 3); % Desired joint angles 1, 2 and 3
b_desired = zeros(discrete_pts, 2); % Desired x and y ball coordinates
dq_desired = zeros(discrete_pts, 3); % Desired joint velocities 1, 2 and 3
b_dot_desired = zeros(discrete_pts, 2); % Desired x and y ball velocites
X_desired = [q_desired b_desired dq_desired b_dot_desired]; % Complete desired state vector

xee_desired = zeros(discrete_pts, 1);
yee_desired = zeros(discrete_pts, 1);
theta_ee_desired = zeros(discrete_pts, 1);
ee_desired = [xee_desired yee_desired theta_ee_desired];

xee_dot_desired = zeros(discrete_pts, 1);
yee_dot_desired = zeros(discrete_pts, 1);
ee_dot_desired = [xee_dot_desired yee_dot_desired];


%% Initial state vector:

q_desired(1,:) = X(1:3);
b_desired(1,:) = X(4:5);
dq_desired(1,:) = X(6:8);
b_dot_desired(1,:) = X(9:10);


%% Finding final joint angles using inverse kinematics:

[q1_temp, q2_temp, q3_temp] = inv_kin(xf_ee, yf_ee, theta_ee_final_oldReference);
q_desired(discrete_pts,1) = q1_temp(1,1); q_desired(discrete_pts,2) = q2_temp(1,1); q_desired(discrete_pts,3) = q3_temp(1,1);
q_desired(discrete_pts,:) = wrapToPi( [(q_desired(discrete_pts,1)+3*pi/2) q_desired(discrete_pts,2) q_desired(discrete_pts,3)] );


%% Desired ball states:

% Computing states of the ball for all time instants in the current bounce
for n = 2:1:discrete_pts
    b_desired(n, 1) = 0;
    b_desired(n, 2) = b_desired(1,2) + b_dot_desired(1,2)*t(n) + (1/2)*(-g)*(t(n)^2);
    b_dot_desired(n, 1) = 0;
    b_dot_desired(n, 2) = b_dot_desired(1,2) + (-g)*(t(n));
end


%% Finding final joint velocities:

% Finding the combination of the final joint velocities (pre-impact) that satisfies the desired post-impact x and y components of the ball velocity
q1_preImpact = q_desired(discrete_pts, 1);
q2_preImpact = q_desired(discrete_pts, 2);
q3_preImpact = q_desired(discrete_pts, 3);
xb_preImpact = b_desired(discrete_pts, 1);
yb_preImpact = b_desired(discrete_pts, 2);
xb_dot_preImpact = b_dot_desired(discrete_pts, 1);
yb_dot_preImpact = b_dot_desired(discrete_pts, 2);

xb_dot_postImpact = xb_dot_postImpact_desired;
yb_dot_postImpact = yb_dot_postImpact_desired;

dq_desired_span = [-5 5];
dq_desired_step = 0.5;

xb_dot_error_threshold = 0.05;
yb_dot_error_threshold = 0.05;

count = 0;
loss = inf;

for dq1_preImpact_temp = [dq_desired_span(1): dq_desired_step: dq_desired_span(2)]
    for dq2_preImpact_temp = [dq_desired_span(1): dq_desired_step: dq_desired_span(2)]
        for dq3_preImpact_temp = [dq_desired_span(1): dq_desired_step: dq_desired_span(2)]
            pre = [q1_preImpact; q2_preImpact; q3_preImpact; xb_preImpact; yb_preImpact; dq1_preImpact_temp; dq2_preImpact_temp; dq3_preImpact_temp; xb_dot_preImpact; yb_dot_preImpact];
            post = impact(pre);
            if( (abs(post(9) - xb_dot_postImpact) <= xb_dot_error_threshold) && (abs(post(10) - yb_dot_postImpact) <= yb_dot_error_threshold) && (sqrt(dq1_preImpact_temp^2 + dq2_preImpact_temp^2 + dq3_preImpact_temp^2) < loss) )
                dq1 = dq1_preImpact_temp;
                dq2 = dq2_preImpact_temp;
                dq3 = dq3_preImpact_temp;

                xb_dot_postImpact_error = post(9) - xb_dot_postImpact;
                yb_dot_postImpact_error = post(10) - yb_dot_postImpact;
                
                loss = sqrt(dq1^2 + dq2^2 + dq3^2); % Minimizing this loss function seemed to give good results since the overall movement of the arm is reduced
                count = count + 1;
            end
        end
    end
end

dq_desired(discrete_pts,1) = dq1; dq_desired(discrete_pts,2) = dq2; dq_desired(discrete_pts,3) = dq3;


%% Finding coefficients for the cubic polynomial trajectory: q(t) = a*(t^3) + b*(t^2) + c*(t) + d

tf = t(discrete_pts,1);
poly_terms = [0        0        0      1;
              tf^3     tf^2     tf     1;
              0        0        1      0;
              3*tf^2   2*tf     1      0];

init_cond = [q_desired(1,1)              q_desired(1,2)              q_desired(1,3);
             q_desired(discrete_pts,1)   q_desired(discrete_pts,2)   q_desired(discrete_pts,3);
             dq_desired(1,1)             dq_desired(1,2)             dq_desired(1,3);
             dq_desired(discrete_pts,1)  dq_desired(discrete_pts,2)  dq_desired(discrete_pts,3)];

traj_coeff = poly_terms \ init_cond;

a = traj_coeff(1, :);
b = traj_coeff(2, :);
c = traj_coeff(3, :);
d = traj_coeff(4, :);


%% Computing desired values:

% The remaining joint angles and joint velocites are computed
for n = 2:1:(discrete_pts-1)
    for i = 1:1:3
        q_desired(n, i) = a(i)*t(n)^3 + b(i)*t(n)^2 + c(i)*t(n) + d(i);
        dq_desired(n, i) = 3*a(i)*t(n)^2 + 2*b(i)*t(n) + c(i);
    end
end

xee_desired = -l1 * sin(q_desired(:,1)) - l2 * sin(q_desired(:,1) + q_desired(:,2)) - l3 * sin(q_desired(:,1) + q_desired(:,2) + q_desired(:,3));
yee_desired = l1 * cos(q_desired(:,1)) + l2 * cos(q_desired(:,1) + q_desired(:,2)) + l3 * cos(q_desired(:,1) + q_desired(:,2) + q_desired(:,3));
theta_ee_desired = q_desired(:,1) + q_desired(:,2) + q_desired(:,3);

xee_dot_desired = dq_desired(:,1) .* ( -l1*cos(q_desired(:,1)) - l2*cos(q_desired(:,1)+q_desired(:,2)) - l3*cos(q_desired(:,1)+q_desired(:,2)+q_desired(:,3)) ) ...
                + dq_desired(:,2) .* ( -l2*cos(q_desired(:,1)+q_desired(:,2)) - l3*cos(q_desired(:,1)+q_desired(:,2)+q_desired(:,3)) ) ...
                + dq_desired(:,3) .* ( -l3*cos(q_desired(:,1)+q_desired(:,2)+q_desired(:,3)) );
yee_dot_desired = dq_desired(:,1) .* ( -l1*sin(q_desired(:,1)) - l2*sin(q_desired(:,1)+q_desired(:,2)) - l3*sin(q_desired(:,1)+q_desired(:,2)+q_desired(:,3)) ) ...
                + dq_desired(:,2) .* ( -l2*sin(q_desired(:,1)+q_desired(:,2)) - l3*sin(q_desired(:,1)+q_desired(:,2)+q_desired(:,3)) ) ...
                + dq_desired(:,3) .* ( -l3*sin(q_desired(:,1)+q_desired(:,2)+q_desired(:,3)) );

X_desired = [q_desired b_desired dq_desired b_dot_desired]; % The complete state vector
ee_desired = [xee_desired yee_desired theta_ee_desired];
ee_dot_desired = [xee_dot_desired yee_dot_desired];


%% Computing errors in desired values:

% These are not being used yet, but could probably be used in the future...
error_in_xee_desired = xee_desired(end) - xf_ee;
error_in_yee_desired = yee_desired(end) - yf_ee;
error_in_theta_ee_desired = theta_ee_desired(end) - theta_ee_final;
% error_in_xee_dot_desired = xee_dot_desired(end) - xf_ee_dot;
error_in_xee_dot_desired = 0;
% error_in_yee_dot_desired = yee_dot_desired(end) - yf_ee_dot;
error_in_yee_dot_desired = 0;

error_in_final_desired = [(error_in_xee_desired), (error_in_yee_desired), (error_in_theta_ee_desired), (error_in_xee_dot_desired), (error_in_yee_dot_desired)];


