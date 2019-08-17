import sounddevice as sd

duration = 5.5  # seconds

from queue import Queue

from threading import Thread

import numpy as np
from numpy import sin, pi
import sounddevice as sd
import time
import matplotlib.pyplot as plt

# Parameters for sine wave
fs = 44100  # sampling rate, Hz
duration = 0.5  # in seconds
f = 15  # sine frequency, Hz


def wave_update(q):
    wave = abs((sin(2 * pi * np.arange(fs * duration) * f / fs)).astype(np.float32))

    print("wave thread started")
    while True:
        if not q.empty():
            gain = q.get()
            sd.play(gain * wave, fs, blocking=True)
            print("q was not empty: gain was", gain)

        else:
            time.sleep(0.5)
            # print("q was empty")


q = Queue(maxsize=0)
q.put(10)
q.put(11)
t = Thread(target=wave_update, args=(q,))
t.start()

for i in range(500):
    q.put(i)
    print("Main thread: sleeping")
    time.sleep(0.5)
