function [A]= read_weights(file_name, dim1,dim2)
    f = fopen(file_name, 'r')
    A = fread(f, [dim1, dim2], 'double')
    