%% Discrete time integration of 2 link arm dynamics
function [tout, xout] = integrateODE(timeSpan, initialState, timeStep, inputTorque)

dt = timeStep;
currentState = initialState;
xout = currentState';
tout = timeSpan(1);

for t = timeSpan(1) + dt:dt:timeSpan(2)
    nextStateDerivative = ode_torque(t, currentState, inputTorque);
    nextVelocity = currentState(6:10) + nextStateDerivative(6:10).*dt;
    nextPosition = currentState(1:5) + nextVelocity.*dt;
    currentState = [nextPosition; nextVelocity];
    xout = [xout; currentState'];
    tout = [tout; t];
end
