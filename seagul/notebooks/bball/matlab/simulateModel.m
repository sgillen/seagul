clc
clear all

initialState = [-pi/4; 3*pi/4; 0.025; 0.5; 0; 0; 0; 0];
timeStep = 0.02;
timeSpan = [0, 1];
n = 2;

for i = 1:1:n
    [tout,xout] = integrateODE(timeSpan, initialState, timeStep,[0;0]);
    [preImpactState, impactTime] = detectImpact(tout,xout);
    
    animate(tout,xout)
    
    postImpactState = impact(preImpactState);
    preImpactState
    postImpactState
    initialState = postImpactState;
    clf
end



