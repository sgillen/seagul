X0 = [-pi/2+.1;0;0;0];
[t,y] = ode45(@acrobot_collocated_linearization,[0 20],X0);
figure(1);
%M = acrobot_animate(t,y);
tau = 0*t;
for n=1:length(t)
    [dx,tau(n)] = acrobot_collocated_linearization(t(n),y(n,:));
end
figure(2); subplot(211); plot(t,tau);


numFeatures = 4;
numHiddenUnits = 12;
numResponses = 1;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)   
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',1500 ... 
)

sigy = std(y)


%ytrain = (y)/sigy
%ytrain = rem(y,pi)

trainedNet = trainNetwork(ytrain',tau',layers,options)