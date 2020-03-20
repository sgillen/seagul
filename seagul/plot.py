"""
Plotting utilities
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Got this from stack overflow


def add_arrow(line, position=None, direction="right", size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == "right":
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate(
        "",
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="-|>", color=color),
        size=size,
    )

def smooth_bounded_curve(
    data,
    time_steps=np.array([]),
    label=None,
    ax=None,
    window=100,
    color='k',
    alpha=.2
):
    """
    Plots the (smoothed) average for many time series plots, as well as plotting the min/max on the same figure
    Designed mostly to make good looking reward curves

    You can either pass in an Axes object you want us to draw on (useful if you are making subplots for example), otherwise we make a new one for you. We return
    whichever Axes we wind up using (as well as the figure, if we make a new one), so you can still modify the plot

    Example:
        from seagul.plot import smooth_bounded_curve

        random_data = np.random.random((1000,10))
        fig, ax = smooth_bounded_curve(random_data)
        ax.set_xlabel("My label") # Can modify whatever you want about the plot afterwards
        plt.show() # not needed in jupyter

    Example:
        from seagul.plot import smooth_bounded_curve

        random_data = np.random.random((1000,10))
        fig, ax = plt.subplot(5,5,figsize=(15,15)
        fig, ax = smooth_bounded_curve(random_data,axs[0,0]) # will only plot on the passed subplot
        ax.set_xlabel("My label") # Can modify whatever you want about the plot afterwards
        plt.show() # not needed in jupyter

    Arguments:
         data: np.array of shape (t,n) where t is the number of timesteps in your data, and n is the number of curves you have
         time_steps: np.array of shape (t,1) containing the timesteps corresponding to the data, if None every datapoint is assigned to one timestep
         label: string containing the name of the model etc. viewed in the legend, if None no legend is displayed
         ax: matplotlib.Axes object to draw the curves on, if None we will make a new object for you
         window: How large of a window to use for the moving average smoothing
         color: what color to make the curve
         alpha: alpha to use for the fillin between min and max values
         time_steps: list or np array labeling the x axis, must be same size as reward curves

    Returns:
        fig: figure object if we created one, else None
        ax: axes object used for plotting
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None

    avg_data = np.zeros(data.shape[0])
    for i in range(data.shape[1]):
        avg_data += data[:,i]

    avg_data /= data.shape[1]
    min_data = [np.min(data[i,:]) for i in range(data.shape[0])]
    max_data = [np.max(data[i,:]) for i in range(data.shape[0])]

    if len(avg_data) > 100:
        min_data = pd.Series(min_data).rolling(100, min_periods=10).mean()
        max_data = pd.Series(max_data).rolling(100, min_periods=10).mean()
        avg_data = pd.Series(avg_data).rolling(100, min_periods=10).mean()

    if time_steps.any() == False:
        time_steps = [i for i in range(data.shape[0])]
    ax.plot(time_steps, avg_data, color=color, label=label)
    ax.fill_between(time_steps, min_data, max_data, color=color, alpha=.2)

    if label != None: # add a legend without multiple labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Average return')
    ax.set_title('Reward curve')

    return fig, ax
