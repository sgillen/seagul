clc
clear all

initialState = [pi/3; -2*pi/3; 2*pi/3; -0.15; 0.6; 0; 0; 0; 0; 0];
timeStep = 0.02;
timeSpan = [0, 1];
n = 2;

for i = 1:1:n
    [tout,xout] = integrateODE(timeSpan, initialState, timeStep);
    [preImpactState, impactTime] = animate(tout,xout);
    postImpactState = impact(preImpactState);
    preImpactState
    postImpactState
    initialState = postImpactState;
    clf
end



