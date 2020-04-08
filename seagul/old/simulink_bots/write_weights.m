function write_weights(file_name, A)
    f = fopen(file_name, 'w');
    fwrite(f,A,'float64');
end

