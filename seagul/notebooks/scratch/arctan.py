import numpy as np
import matplotlib.pyplot as plt

x = np.random.random((10,1)) # same as "a" in your example
y = np.random.random((10,1)) # same as "b" in your example

plt.plot(x,y)
plt.title("Rectangular coordinates")
plt.show()

th = np.arctan2(y,x)
r = np.sqrt(x**2 + y**2)

plt.polar(th,r)
plt.title("Polar coordinates")
plt.show()
