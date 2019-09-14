import numpy as np

import math
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import (BasicVector, DiagramBuilder, FloatingBaseType,
                         RigidBodyPlant, RigidBodyTree, Simulator,SignalLogger, 
                         Isometry3, DirectCollocation, PiecewisePolynomial, VectorSystem)
from pydrake.solvers.mathematicalprogram import Solve

from pydrake.attic.multibody.shapes import VisualElement, Box
from pydrake.attic.multibody.collision import CollisionElement

#from pydrake.all import 
from underactuated import (FindResource, PlanarRigidBodyVisualizer)

from IPython.display import HTML
import matplotlib.pyplot as plt

#this one is home grown, make sure it's in the same directory as this notebook
from seagul.drake import x_expr,y_expr,x_float,y_float,x_taylor,y_taylor
from seagul.resources import getResourcePath 
from numpy import pi

## ===========================

from gym import core, spaces
from gym.utils import seeding


def DrakeWalkerEnv(core.env):

    def __init__():
        pass

    def reset():
        
    

        
