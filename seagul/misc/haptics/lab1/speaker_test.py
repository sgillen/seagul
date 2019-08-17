import numpy as np
from numpy import sin, pi
import sounddevice as sd
import time
import matplotlib.pyplot as plt

# Parameters for sine wave
fs = 44100  # sampling rate, Hz
duration = 60.0  # in seconds
f = 15  # sine frequency, Hz

wave = abs((sin(2 * pi * np.arange(fs * duration) * f / fs)).astype(np.float32))

gain = 10
sd.play(gain * wave, fs)

# plt.plot(wave[1:10000])
plt.show()

while True:
    time.sleep(5)
