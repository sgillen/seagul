# force keras to use the cpu becuase it's actually faster for this size network

#.import os
import numpy as np
import pandas as pd
from numpy import cos, sin, pi
import matplotlib.pyplot as plt
import scipy.integrate as integrate


#import matplotlib.animation as animation
#from ipython.display import html
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Flatten
from tensorflow.keras.constraints import max_norm
from cartpole_class import Cartpole


def sampleData(x, n, Ts):
    """spliTs up data array x into seTs of length n with sampling time Ts
    x:  input matrix (M,N)
    n:  number of samples per set
    Ts: sample time
    """
    N = x.shape[0]                                              #number of input vectors
    z_ext = np.zeros(((n-1)*Ts,x.shape[1]))
    x = np.concatenate((z_ext, x), axis=0)
    #calculate number of seTs
    nset = N
    y = np.zeros((nset,)+(x.shape[1],)+(n,))                    #initialize output matrix
    step = 0      
    #iterate through input data                                              
    while(step<nset):
        #select vectors according to n and Ts
        y[step,:,:] = np.transpose(x[step:(n-1)*Ts+1+step:Ts,:]) 
        step+=1                                                 
    return y

# define some constanTs


# time vector, doesn't actually affect the integration, only what times it records our state variable at
dt = 0.01
time = np.arange(0.0, 20, dt)

# cartpole is a class we defined that takes care of the simulation/animation of the cartpole
bot = Cartpole(dt,[],[])

# parameters for the amount of different trajectories we generate with the energy shaping controller
num_trials = 1
num_states = 4
num_t = len(time)
y = np.zeros((num_t, num_states, num_trials))

for i in range(num_trials):
    # initial conditions
    theta = 0
    x = 0.0
    th_dot = 2*(i/num_trials) - 1 
    xdot = 0.0

    # initial state
    init_state = np.array([theta, x, th_dot, xdot])

    # integrate the ode using scipy.integrate.
    # todo switch over to the more modern solve_ivp, as we do for the pendubot
    u_hist = []
    y[:, :, i] = integrate.odeint(bot.derivs, init_state, time)


u = np.zeros((num_t, num_trials))

for i in range(num_trials):
    for t in range(num_t):
        u[t,i] = bot.control(y[t,:,i]) 
        
# create our two rnns for comparison

look_back = 3
Ts = 1
ys = sampleData(y[:,:,0], look_back, Ts)
#us = sampledata2(u[:,0,np.newaxis], 1, 1)
#us = us[:,:,0:]
# feed forward with look-back 
with tf.variable_scope('pi/simple_pol/'):
    fflb_model = Sequential()
    fflb_model.add(Flatten())
    fflb_model.add(Dense(12*look_back, activation ='relu'))
    fflb_model.add(Dense(12*look_back, activation ='relu'))
    fflb_model.add(Dense(1))

fflb_model.compile(loss='mean_squared_error', optimizer='adam')
    
# train feedforward network

fflb_history = fflb_model.fit(ys, u, batch_size = ys.shape[0], epochs=3000, verbose=1)
plt.plot(fflb_history.history['loss'])
plt.title('simple model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.figure()


# an ugly hack todo make this and the one above compatible
def make_fflb_controller(model):
    def nn_controller(q):
        if (q[0,look_back-1] < (140 * (pi/180)) ) or (q[0,look_back-1] > (220 * (pi/180)) ):
            u = model.predict(q.reshape(1,q.shape[0],q.shape[1]))
            return u[0,0]
        else:
            # balancing
            # lqr: k values from matlab
            k1 = 140.560
            k2 = -3.162
            k3 = 41.772
            k4 = -8.314
            u = -(k1 * (q[0,look_back-1] - pi) + k2 * q[1,look_back-1] + k3 * q[2,look_back-1] + k4 * q[3,look_back-1])
            return u
        
    return nn_controller


fflb_control = make_fflb_controller(fflb_model)

fflb_bot = Cartpole(dt,Ts,look_back)

fflb_bot.control = fflb_control

# initial conditions
theta = .4
x = 1
th_dot = .1 # an initial velocity, triggers the swing up control
xdot = 0.1
time = np.arange(0.0, 20, dt)

# initial state
init_state = np.array([theta, x, th_dot, xdot])

# run the simulation for the feedforward network

# Fill in our u after the fact..
y_fflb = integrate.odeint(fflb_bot.derivs_dig_lb, init_state, time)
#u_fflb = np.zeros((ys.shape[0],1))

#for i in range(ys.shape[0]):
       # u_fflb[i] = fflb_bot.control(y[i,:,-1]) 

plt.figure()
plt.plot(y_fflb[:,0])
#%matplotlib qt
ani = fflb_bot.animate_cart(time, y_fflb)
#HTML(ani.to_jshtml())
