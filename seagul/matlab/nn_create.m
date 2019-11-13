input_size = 4
output_size = 1
layer_size = [64,64]

sample_x = randn(input_size, 1)
sample_y = randn(output_size,1)

nn = feedforwardnet(layer_size)
nn = configure(nn, sample_x, sample_y)
nn = init(nn)

fileID = fopen('layer1.bin','w');
fwrite(fileID,magic(4),'double');

%a = memmapfile('weights.sg')