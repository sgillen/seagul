%poor name in hindsight, this takes a battery serial number and gives a
%corresponding 'bin' to store it in. nothing to do with binary

%this is about to be hyper janky but there isn't any pattern I can discern
%from the serial #s, and the address field changes which battery is which
%sometimes.

function bin = serial_to_bin(serial)
    switch(serial)
        case 9132
            bin = 1; %1.1
        case 9259
            bin = 2; %1.2
        case 9312
            bin = 3; %1.3
        case 9316
            bin = 4; %1.4
        case 9233
            bin = 5; %1.5
        case 9266
            bin = 6; %2.1
        case 9107
            bin = 7; %2.2
        case 9229
            bin = 8; %2.3
        case 9247
            bin = 9; %2.4
        case 9230
            bin = 10; %2.5
        case 9273
            bin = 11; %3.1
        case 9258
            bin = 12; %3.2
        case 9150
            bin = 13; %3.3
        case 9318
            bin = 14; %3.4
        case 9257
            bin = 15; %3.5
        case 9231
            bin = 16; %4.1
        case 9091
            bin = 17; %4.2
        case 9246
            bin = 18; %4.3
        case 9112
            bin = 19; %4.4
        case 9274
            bin = 20; %4.5
        otherwise 
            error('serial number %i does not match any known battery',serial)
    end
      
    
    
        