import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from numba import jit
import time


class Cartpole:
    """
    Implements dynamics, animation, and control for for a simple cartpole pendulum.

    Meant as a testbed for different controllers, the default controller (implemented in control) does a pretty good job
    though.

    The default constructor just goes, I find it more convenient to just go ahead and construct an object and then c
    change parameters after the fact.

    I.E.

    cart = Cartpole()
    cart.L = 5.0

    Attributes:
        L - length of the pendulum in (m)
        mc - mass of the kart (kg)
        mp - magnitude of pointmass at the end of the cart's pole (kg)
        g - force f gravity (N)
    
    """

    # Define constants (geometry and mass properties):
    def __init__(self, dt=None, Ts=None, n=None):
        self.L = 1.0;  # length of the pole (m)
        self.mc = 4.0  # mass of the cart (kg)
        self.mp = 1.0  # mass of the ball at the end of the pole

        self.g = 9.8;
       
        self.Ts = Ts
        self.n = n;
        self.dt = dt
        self.tNext = 0
        self.u_hold = []
        self.y_lb = []

    def animate_cart(self, t, y):
        """
        constructs an animation object and returns it to the user.

        Then depending on your environment you'll need to do some other call to actually display the animation.
        usually I'm calling this from a jupyter notebook, in which case I do:



        ani = bot.animate_cart(time, y)
        HTML(ani.to_jshtml())


        Args:
            t: numpy array with the time steps corresponding to the trajectory you want to animate, does not have to
            be uniform

            y: numpy array with a trajectory of state variables you want animated. [theta , x, thetadot, xdot]

        Returns:
            matplotlib.animation, which you then need to display
        """

        dt = (t[-1] - t[0])/len(t)


        x1 = y[:, 1]
        y1 = 0.0

        x2 = self.L * sin(y[:, 0]) + x1
        y2 = -self.L * cos(y[:, 0]) + y1

        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, aspect='equal',
                             xlim=(-3, 3), ylim=(-3, 3))
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=2)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text


        def animate(i):
            thisx = [x1[i], x2[i]]
            thisy = [y1, y2[i]]

            line.set_data(thisx, thisy)
            time_text.set_text(time_template % (i * dt))
            return line, time_text

        return animation.FuncAnimation(fig, animate, np.arange(1, len(y)), interval=40, blit=True, init_func=init)

    # @jit(nopython=False)
    def control(self, t, q):
        """
        This is where you should define the control for the cartpole, called by derivs.

        By default, implements a swingup controller for the cartpole based on energy shaping. Switches to an LQR to
        balance the pendulum

        Args:
            t: float with the current time step (may not be used)
            q: numpy array of state variables [theta, x, thetadot, xdot]

        Returns:
            u, the control torque in N*m
        """


        # Ausutay Ozmen
        if (q[0] < 140 * pi/180) or (q[0] > 220 * pi/180 ):
            # swing up
            # energy error: Ee
            Ee = 0.5 * self.mp * self.L * self.L * q[2] ** 2 - self.mp * self.g * self.L * (1 + cos(q[0]))
            # energy control gain:
            k = 0.23
            # input acceleration: A (of cart)
            A = k * Ee * cos(q[0]) * q[2]
            # convert A to u (using EOM)
            delta = self.mp * sin(q[0]) ** 2 + self.mc
            u = A * delta - self.mp * self.L * (q[2] ** 2) * sin(q[0]) - self.mp * self.g * sin(q[2]) * cos(q[2])
        else:
            # balancing
            # LQR: K values from MATLAB
            k1 = 140.560
            k2 = -3.162
            k3 = 41.772
            k4 = -8.314
            u = -(k1 * (q[0] - pi) + k2 * q[1] + k3 * q[2] + k4 * q[3])
        return u

    # state vector: q = transpose([theta, x, d(theta)/dt, dx/dt])
    # @jit(nopython=False)
    
    def derivs(self, t, q):
        """
        Implements the dynamics for our cartpole, you need to integrate this yourself with E.G:
        y = integrate.odeint(bot.derivs, init_state, time)
        or whatever other ode solver you prefer.


        Args:
            t: float with the current time (not actually used but most ODE solvers want to pass this in anyway)
            q: numpy array of state variables [theta, x, thetadot, xdot]
            numpy array with the derivatives of the current state variable [thetadot, xdot, theta2dot, x2dot]

        Returns:
            dqdt: numpy array with the derivatives of the current state variable [thetadot, xdot, theta2dot, x2dot]
        """

        dqdt = np.zeros_like(q)

        # control input
        u = self.control(t, q)

        delta = self.mp * sin(q[0]) ** 2 + self.mc

        dqdt[0] = q[2]
        dqdt[1] = q[3]

        dqdt[2] = - self.mp * (q[2] ** 2) * sin(q[0]) * cos(q[0]) / delta \
                  - (self.mp + self.mc) * self.g * sin(q[0]) / delta / self.L \
                  - u * cos(q[0]) / delta / self.L

        dqdt[3] = self.mp * self.L * (q[2] ** 2) * sin(q[0]) / delta \
                  + self.mp * self.L * self.g * sin(q[0]) * cos(q[0]) / delta / self. L \
                  + u / delta
                  
        return dqdt
    
    def derivs_dig(self, t, q):
        """
        Implements the dynamics for our cartpole, you need to integrate this yourself with E.G:
        y = integrate.odeint(bot.derivs, init_state, time)
        or whatever other ode solver you prefer.

        This version only updates the control input at fixed intervals (instead of every time the solver is updates)



        Args:
            t: float with the current time (not actually used but most ODE solvers want to pass this in anyway)
            q: numpy array of state variables [theta, x, thetadot, xdot]
            numpy array with the derivatives of the current state variable [thetadot, xdot, theta2dot, x2dot]

        Returns:
            dqdt: numpy array with the derivatives of the current state variable [thetadot, xdot, theta2dot, x2dot]
        """

        if(t>=self.tNext):    #<>
            self.tNext += self.Ts*self.dt
            self.u_hold = self.control(q)
            
        dqdt = np.zeros_like(q)
        
        delta = self.mp * sin(q[0]) ** 2 + self.mc
    
        dqdt[0] = q[2]
        dqdt[1] = q[3]
        
        dqdt[2] = - self.mp * (q[2] ** 2) * sin(q[0]) * cos(q[0]) / delta \
                      - (self.mp + self.mc) * self.g * sin(q[0]) / delta / self.L \
                      - self.u_hold * cos(q[0]) / delta / self.L
    
        dqdt[3] = self.mp * self.L * (q[2] ** 2) * sin(q[0]) / delta \
                  + self.mp * self.L * self.g * sin(q[0]) * cos(q[0]) / delta / self. L \
                  + self.u_hold / delta
  
        return dqdt

    # Marco Molnar
    def derivs_dig_lb(self, t, q):
        """
        Implements the dynamics for our cartpole, you need to integrate this yourself with E.G:
        y = integrate.odeint(bot.derivs, init_state, time)
        or whatever other ode solver you prefer.

        This version only updates the control input at fixed intervals (instead of every time the solver is updates).
        Furthermore this version keeps track of previous inputs so we can use it with a lookback mlp


        Args:
            t: float with the current time (not actually used but most ODE solvers want to pass this in anyway)
            q: numpy array of state variables [theta, x, thetadot, xdot]
            numpy array with the derivatives of the current state variable [thetadot, xdot, theta2dot, x2dot]

        Returns:
            dqdt: numpy array with the derivatives of the current state variable [thetadot, xdot, theta2dot, x2dot]
        """

        if(t==0):
            z_ext = np.zeros((q.size,(self.n-1)*self.Ts))
            self.y_lb = np.concatenate((z_ext, q[:,np.newaxis]), axis=1) 
            self.tNext = self.Ts*self.dt
            self.u_lb = self.control(self.y_lb)
        else:
            if(t>=self.tNext):    #<>
                self.tNext += self.Ts*self.dt
                self.y_lb = np.concatenate((self.y_lb[:,1:],q[:,np.newaxis]), axis=1)
                self.u_lb = self.control(self.y_lb)
                
        dqdt = np.zeros_like(q)
    
        delta = self.mp * sin(q[0]) ** 2 + self.mc

        dqdt[0] = q[2]
        dqdt[1] = q[3]
        
        dqdt[2] = - self.mp * (q[2] ** 2) * sin(q[0]) * cos(q[0]) / delta \
                      - (self.mp + self.mc) * self.g * sin(q[0]) / delta / self.L \
                      - self.u_lb * cos(q[0]) / delta / self.L
    
        dqdt[3] = self.mp * self.L * (q[2] ** 2) * sin(q[0]) / delta \
                  + self.mp * self.L * self.g * sin(q[0]) * cos(q[0]) / delta / self. L \
                  + self.u_lb / delta
  
        return dqdt

    # Marco Molnar
    def animate_cart_dim(self, t, Y, LABEL_ROWS,LABEL_COLS, info):
        """
        constructs an animation object and returns it to the user.

        Then depending on your environment you'll need to do some other call to actually display the animation.
        usually I'm calling this from a jupyter notebook, in which case I do:

        ani = bot.animate_cart(time, t, Y, LABEL_ROWS,LABEL_COLS, info)
        HTML(ani.to_jshtml())


        Args:
            t: numpy array with the time steps corresponding to the trajectory you want to animate, does not have to
            be uniform

            Y: 4D-Numpy matrix with numpy arrays containing trajectories of state variables you want animated.
            Last two dimensions for selecting the numpy array trajectories [theta , x, thetadot, xdot] of dimension [Nx4] with N
            as the number of trajectory samples.

            LABEL_ROWS, LABEL_COLS: Row and Column labels for subplots grid

            info: 2D list with strings for having an info text at the respective position in the subplots

        Returns:
             matplotlib.animation, which you then need to display
        """
        #Convert row and col to linear index
        def sub2ind(array_shape, rows, cols):
            return rows*array_shape[1] + cols +1
        
        dt = (t[-1] - t[0])/len(t)              #Get time step
        dim_sub = Y.shape[2:4]                  #shape of subplot grid
        N = Y.shape[0]                          #Number of samples
        Nplot = dim_sub[0]*dim_sub[1]           #Number of subplots
        
        #Initialize variables plot points, axes, lines and text
        X1 = np.zeros((N, dim_sub[0], dim_sub[1]))
        Y1 = 0
        X2 = np.zeros((N, dim_sub[0], dim_sub[1]))
        Y2 = np.zeros((N, dim_sub[0], dim_sub[1]))
        AX = [[0 for x in range(dim_sub[1])] for y in range(dim_sub[0])] 
        LINE = [0 for x in range(Nplot)]
        TIME_TEXT = [0 for x in range(Nplot)]
        time_template = 'time = %.1fs'    
        
        fig = plt.figure()
        
        #Iterate trough all trajectories and create plot points, subplots, info texts and time
        for i in range(0,dim_sub[0]):
            for j in range(0,dim_sub[1]):
                #Trajectories to plot points
                X1[:,i,j] = Y[:,1,i,j]
                X2[:,i,j] = self.L * sin(Y[:,0,i,j]) + X1[:,i,j]
                Y2[:,i,j] = -self.L * cos(Y[:,0,i,j]) + Y1
                #Subplots
                AX[i][j]  = fig.add_subplot(dim_sub[0], dim_sub[1], sub2ind(dim_sub,i,j), autoscale_on=False, aspect='equal',
                             xlim=(-3, 3), ylim=(-3, 3))
                AX[i][j].grid()
                #Labels for columns and rows
                if i==0:
                    AX[i][j].set_title(LABEL_COLS[j])
                if j==0:
                    AX[i][j].text(-0.2, 0.55, LABEL_ROWS[i], transform=AX[i][j].transAxes, rotation=90)
                #Create line objects
                LINE[sub2ind(dim_sub,i,j)-1], = AX[i][j].plot([], [], 'o-', lw=2)
                #Add info text to plot
                AX[i][j].text(0.05, 0.1, info[i][j], transform=AX[i][j].transAxes)
                #Add time text to plot
                TIME_TEXT[sub2ind(dim_sub,i,j)-1] = AX[i][j].text(0.05, 0.9, '', transform=AX[i][j].transAxes)
        #Append text objects to line objects (necessary for using matplotlib FuncAnimation)         
        LINE += TIME_TEXT
        #Init for FuncAnimation
        def init():
            
            for i in range(0,dim_sub[0]):
                for j in range(0,dim_sub[1]):
                    LINE[sub2ind(dim_sub,i,j)-1].set_data([],[])
                    LINE[Nplot+sub2ind(dim_sub,i,j)-1].set_text('')
            return LINE
        #During animation adapt plot and time text
        def animate(k):

            for i in range(0,dim_sub[0]):
                for j in range(0,dim_sub[1]):
                    thisx = [X1[k,i,j], X2[k,i,j]]
                    thisy = [Y1,Y2[k,i,j]]
                    LINE[sub2ind(dim_sub,i,j)-1].set_data(thisx, thisy)
                    LINE[Nplot+sub2ind(dim_sub,i,j)-1].set_text(time_template % (k * dt))
            return LINE

        return animation.FuncAnimation(fig, animate, np.arange(1, N), interval=5, blit=True, init_func=init)
    
