function [ y ] = ffnn(x) % might make these global in a second...
%fully connected network
global w
global b
num_layers = size(w,2);

for i = 1:num_layers-1
    x = w{i}'*x + b{i};
    x = tanh(x);
end

y = w{end}'*x;
end