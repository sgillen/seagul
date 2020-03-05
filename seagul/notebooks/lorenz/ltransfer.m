% This comes from the code used to transfer weights to the simulink bots 
% In that setting you can't iterate through cell arrays, and you can't
% call external functions, that's why this code is gross.

dtype = 'float32';
hsize = 32;
in_size = 4;
out_size = 2;
num_weights = (hsize^2) + hsize*9;

net_dir = 'euler_net/';
weight_names = {'wght0.dat', 'wght1.dat', 'wght2.dat'};
bias_names   = {'bias0.dat', 'bias1.dat', 'bias2.dat'};

w = {single(rand(in_size,hsize)/sqrt(num_weights)), single(rand(hsize,hsize)/sqrt(num_weights)), single(rand(hsize,out_size*2)/sqrt(num_weights))};
b = {single(rand(hsize,in_size)/sqrt(num_weights)), single(rand(hsize,1)/sqrt(num_weights)), single(rand(out_size*2,1)/sqrt(num_weights))};

f = fopen(strcat(net_dir, weight_names{1}), 'r'); w{1} = fread(f, [hsize,in_size], dtype); fclose(f);
f = fopen(strcat(net_dir, weight_names{2}), 'r'); w{2} = fread(f, [hsize, hsize], dtype); fclose(f);
f = fopen(strcat(net_dir, weight_names{3}), 'r'); w{3} = fread(f, [out_size*2,hsize], dtype); fclose(f);

f = fopen(strcat(net_dir, bias_names{1}), 'r'); b{1} = fread(f, [1, hsize], dtype); fclose(f);
f = fopen(strcat(net_dir, bias_names{2}), 'r'); b{2} = fread(f, [1, hsize], dtype); fclose(f);
f = fopen(strcat(net_dir, bias_names{3}), 'r'); b{3} = fread(f, [1,out_size*2], dtype); fclose(f);

x = [1,2,3,4];
x1 = tanh(x*w{1}' + b{1});
x2 = tanh(x1*w{2}' + b{2});
out = x2*w{3}' + b{3};
means = out(1:2);
std = out(3:4);