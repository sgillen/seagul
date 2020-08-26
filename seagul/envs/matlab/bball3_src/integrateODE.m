%% Discrete time integration of 3 link arm dynamics

function [tout, xout, u] = integrateODE(t, X_desired)

discrete_pts = length(t); % Number of discretized instants of time
dt = t(2) - t(1); % Computation time step

currentState = X_desired(1,:)';

xout(1, :) = currentState;
tout(1) = t(1);
u(1, :) = [0 0 0];

for n = 2:1:discrete_pts
    X_des = X_desired(n, :)';
    [nextStateDerivative, U] = ode_torque(t(n), currentState, X_des);
    nextVelocity = currentState(6:10, :) + nextStateDerivative(6:10, :) * dt;
    nextPosition = currentState(1:5, :) + nextVelocity * dt;
    currentState = [nextPosition; nextVelocity];
    xout(n, :) = currentState';
    tout(n) = t(n);
    u(n, :) = U';
end

