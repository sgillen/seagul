%% Remus RLF decoder

%!! This code assumes you've already imported the bat.mat file you're interested in,
% we could change that easily of course.

%!! This code also depends on a little function I wrote called
%serial_to_bin, just make sure you put it into the same directory as this
%script

%% Summary
%{
This code will take data exported from the VIP's data exporter and make it
a little more human readable. The result should be 5 20x<time>x<#missions>
arrays of information. each of the 20 bins in the first dimension
represent one of the batteries. the second dimension is the number of
polls the computer has done since the start of the mission, I think it's best
to think of it as a timestep. the last dimension is the mission number 
(it should be a singleton dimension unless you give it a rlf file with more
than one mission)

%}
%% Example
%{
so as an example my_status(1,20,2) will give you the status of the first
battery (1.1) at timestep 20 in mission 2.

The easiest way I've found to
view these is just to have matlab print them out, it will print my_x(:,:,1)
m_x(:,:,2) sequentially, each collumn represents an array of all the
batteries, each row is another timestep. 

hopefully the annotations below help.. an empty cell (or zero for double
fields) means that battery was missing. 

my_status(:,:,4) =  <- printing mission 4

 Columns 1 through 13
     time 1  time 2   time 3  ... 
1.1 -> []    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'
1.2 -> []    'C0'     'C0'     'C0'     'C0'     'C0'     'C0'     'C0'     'C0'     'C0'     'C0'     'C0'     'C0' 
etc -> []    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'
       []    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'
    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'
       []    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'
       []    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'
       []    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'
       []    'C0'     'C0'     'C0'     'C0'     'C0'     'C0'     'C0'     'C0'     'C0'     'C0'     'C0'     'C0' 
    'C0'     '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'
       []    '1C0'    '1C0'    'C0'     'C0'     'C0'     'C0'     'C0'     '1C0'    'C0'     'C0'     'C0'     '1C0'
       []       []       []       []       []       []       []       []       []       []       []       []       []
       []    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'
    'C0'     'C0'     'C0'     'C0'     'C0'     'C0'     'C0'     '1C0'    '1C0'    'C0'     'C0'     'C0'     '1C0'
       []    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'
       []    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'
       []    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'
       []    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'
       []    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    'C0'     '1C0'    '1C0'
    'C0'        []    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    '1C0'    'C0'     '1C0'


%}

dt = diff(secs_since_1970);
time_tolerance = 30; %a little arbitrary, but it seems to work. 
mission_time_tolerance = 120; %again, arbitrary but seemed to work

mission = 1;
count = 1;

%if you don't itialize time_s as a datetime matlab throws a fit. 
time_s = datetime([],[],[]);

%could initialize the remaining values to zero arrays but the script
%is already fast enough and this way is easier, if we start needing to
%process very large files I might need to change this.

for i = 1:size(dt,2) %dt is size(secs_since_1970) - 1 so we go to that range instead.
    %if there's been enough time between polls, consider the next poll as
    %the start of a new mission, and reset the number of polls.
    if(dt(i) > mission_time_tolerance)
        mission = mission + 1;
        count = 1;
    %if we're not just starting the mission and the time between the last
    %poll and now is large we're in the next sweep so increase count by
    %one.
    elseif(dt(i) > time_tolerance)
        count = count + 1;
    end
    
    %no matter what record the status/flags/time of the data point we're at
    %and put it in the correct address bin. 
    
    my_serial(serial_to_bin(serial_number(i)), count, mission) = serial_number(i);
    my_status(serial_to_bin(serial_number(i)), count, mission) = cellstr(dec2hex(battery_status(i)));
    my_flags(serial_to_bin(serial_number(i)), count,mission) = cellstr(dec2hex(battery_flags(i)));
    my_time(serial_to_bin(serial_number(i)), count, mission) = secs_since_1970(i);
    my_time_s(serial_to_bin(serial_number(i)), count, mission) = datetime(secs_since_1970(i),'ConvertFrom','posixtime');
   
    %can add arbitrarily more fields if we need to.
   
   
    
end



