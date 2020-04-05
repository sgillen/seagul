from seagul.integration import euler
from seagul.integrationx import euler as eulerx
#from seagul.integrationx import euler32
import time

from seagul.envs.simple_nonlinear.linear_z import LinearEnv
from seagul.envs.simple_nonlinear.linear_zx import LinearEnvX

cimport numpy as np
DTYPE = np.float32


def dynamics(t,q,u):
    return q + u

cdef int num_steps = int(1e5)
def main2():
    s = 0
    start = time.time()
    for _ in range(num_steps):
        s = euler(dynamics,1,0,.1,s)
    print(time.time() - start)

    s = 0
    start = time.time()
    for _ in range(num_steps):
        s = eulerx(dynamics,1,0,.1,s)
    print(time.time() - start)

    cdef float s2 = 0
    start = time.time()
    for _ in range(num_steps):
        s2 = eulerx(dynamics,1,0,.1,s2)
    print(time.time() - start)
    

def main():
    env = LinearEnv()
    start = time.time()
    x = np.array([0,0])
    for _ in range(num_steps):
        env.step(x)
    print(time.time() - start)

    env2 = LinearEnvX()
    start = time.time()

    for _ in range(num_steps):
        env2.step(x)
    print(time.time() - start)
    
    
if __name__ == "__main__":
    main()

# s = 0
# start = time.time()
# for _ in range(1000):
#     s = euler(0,s,1)
# print(time.time() - start)
