#This script takes a list of waypoints described in a config file specified by the command line (see config.py for an example)
#this was written out of boredom during a long and boring in water test of an AUV

import datetime
import argparse

parser = argparse.ArgumentParser(description='converts a list of points to a REMUS rmf file')
parser.add_argument('input_file')
parser.add_argument('output_file')

args = parser.parse_args()
print args.input_file

#output_file = 'test.txt'
#input_file = 'in.txt'

ofp = open('test.txt', 'w')

print 'opened file, writing header'
#Write the header
ofp.write('# Location: ' + location + '\n')
ofp.write('# Version: ' + str(datetime.datetime.now()) + '\n')


#try and write moos paramaters, may not always be required
if 'lat0' in locals():
    ofp.write('# LatOrigin: ' + lat0 + '\n')
else:
    print 'Warning: no LatOrigin set'

if 'lon0' in locals():
    ofp.write('# Longorigin: '+ lon0 + '\n\n') 
else:
    print 'Warning: no LonOrigin set'


print 'printing waypoints'

for point in points:
    [label, latlon] = point.split(':')
    [lat, lon] = latlon.split(',')
    
    lon.lstrip('-') #we assume we're operating in the western hemisphere

    ofp.write('[Location]\n')
    ofp.write('Type=Waypoint \n')
    ofp.write('Label=' + label + '\n')
    ofp.write('Position=' + lat + 'N' + lon + 'W\n\n')

print 'waypoints written'
