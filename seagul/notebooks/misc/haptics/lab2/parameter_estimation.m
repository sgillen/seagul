%Neeli Tummala and Paige Sullivan

first_peak = [3.341, 0.03133-0.003]; % first data point
second_peak = [3.709, 0.01895-0.003]; % second data point
% the 0.003 is subtracted because steady state is 0.003, not 0
x0 = first_peak(2); % amplitude of first peak
x1 = second_peak(2); % amplitude of second peak
t0 = first_peak(1); % time of first peak
t1 = second_peak(1); % time of second peak
k = 500; % Spring constant used: 500 N/m

T = t1-t0; % period of oscillation in seconds
F = 1/T; % frequency of oscillation in Hz
alpha = log(x0/x1) / (2*pi); % unitless damping constant
omega = (2*pi*F) / sqrt(1-alpha^2); % rad/s
m = k / omega^2; % kg