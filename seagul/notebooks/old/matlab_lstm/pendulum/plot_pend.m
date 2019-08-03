% script to plot data from  "pendulum swingup.ipynb". Only used because
% python will sometimes choke when zooming in to 

data = data1000;
hold on

for i = 1:size(data(1,:,1,1),2)
    for j = 1:size(data(1,1,:,1),3)
        if data(end,i,j,1) > (-.1+ pi) && data(end,i,j,1) < (.1 + pi)
            plot(data(:,i,j,1), data(:,i,j,2), 'x-');
        end
    end
end

grid on