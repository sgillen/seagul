import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.signal import sawtooth
from numpy import sin, pi
import sounddevice as sd


# Open the default camera (should be your built in webcam if you have one)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("something went wrong! video not open")
    raise SystemExit

# Define parameters that the marker detection library needs
DICTIONARY = aruco.DICT_6X6_1000
aruco_dict = aruco.Dictionary_get(DICTIONARY)
aruco_parameters = aruco.DetectorParameters_create()

# make a window that will display all the found markers
# re, img = cap.read()
# cv2.namedWindow('Markers')
# cv2.imshow('Markers', img)

# Parameters for the wave we send to the motor
fs = 44100  # sampling rate, Hz
duration = 0.5  # in seconds
f = 15  # sine frequency, Hz

wave = abs((sin(2 * pi * np.arange(fs * duration) * f / fs)).astype(np.float32))
# wave = sawtooth(2*pi*np.arange(fs*duration)*f/fs).astype(np.float32)
wave_play = wave.copy()  # Make a copy of the wave for the modified version that actually gets sent to the motor

baseline = 70
static_gain = 0.01

# Start the game loop
while True:

    # run_util the marker detection
    re, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_parameters)

    # found = aruco.drawDetectedMarkers(img, corners, ids)
    # cv2.imshow('Markers', found)

    # as long as we found at least one marker, go ahead and change the amplitude
    if corners:
        tracked_marker = corners[0].squeeze()
        # This is the x position of opposing corners of the marker (in pixels). Subtracting the two is a lazy way to
        # estimate the distance of the marker from the screen
        x_track1 = tracked_marker[0, 0]
        x_track2 = tracked_marker[2, 0]
        gain = (abs(x_track1 - x_track2) - baseline) * static_gain

        # Modify the played waveform we play (just change the amplitude for now)
        wave_play = gain * wave

        print(abs(x_track1 - x_track2))

    # Go ahead and play the wave regardless if we updated the position of the marker
    sd.play(wave_play, fs, blocking=True)

    # give us a way to actually quit
    if cv2.waitKey(1) == ord("q"):
        break

sd.stop
cap.release()
