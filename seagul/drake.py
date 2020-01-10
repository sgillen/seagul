import pydrake.symbolic as sym
import math


def y_expr(state):
    # type: (List[float])->list[float]
    """ Return the y coordinate of the drake_walkers toe as an expression
        I.E. using pydrake.cos
        TODO more documentation
    """
    y1_in = sym.cos(state[0])
    y2_in = y1_in + sym.cos(state[0] + state[1])
    y4_in = y2_in + sym.cos(state[0] + state[1] + state[3])
    y5_in = y4_in + sym.cos(state[0] + state[1] + state[3] + state[4])

    return -y5_in


def x_expr(state):
    # type: (List[float])->list[float]
    """ Return the x coordinate of the drake_walkers toe as an expression
        I.E. using pydrake.sin
        TODO more documentation
    """
    x1_in = sym.sin(state[0])
    x2_in = x1_in + sym.sin(state[0] + state[1])
    x4_in = x2_in + sym.sin(state[0] + state[1] + state[3])
    x5_in = x4_in + sym.sin(state[0] + state[1] + state[3] + state[4])

    return -x5_in


def y_expr_prism(state):
    # type: (List[float])->list[float]
    """ Return the y coordinate of the prism (floating base( walker as a float  
        I.E. using sym.cos 
        TODO more documentation
    """
    y1_in = sym.cos(state[2])
    y2_in = y1_in + sym.cos(state[2] + state[3])
    y4_in = y2_in + sym.cos(state[2] + state[3] + state[5])
    y5_in = y4_in + sym.cos(state[2] + state[3] + state[5] + state[6])

    return -y5_in


def x_expr_prism(state):
    # type: (List[float])->list[float]
    """ 
        Return the x coordinate of the walker as a float  
        I.E. using sym.sin
        TODO more documentation
    """
    x1_in = sym.sin(state[2])
    x2_in = x1_in + sym.sin(state[2] + state[3])
    x4_in = x2_in + sym.sin(state[2] + state[3] + state[5])
    x5_in = x4_in + sym.sin(state[2] + state[3] + state[5] + state[6])

    return -x5_in


def y_float(state):
    # type: (List[float])->list[float]
    """ Return the y coordinate of the walker as a float  
        I.E. using math.cos 
        TODO more documentation
    """
    y1_in = math.cos(state[0])
    y2_in = y1_in + math.cos(state[0] + state[1])
    y4_in = y2_in + math.cos(state[0] + state[1] + state[3])
    y5_in = y4_in + math.cos(state[0] + state[1] + state[3] + state[4])

    return -y5_in


def y_float_prism(state):
    # type: (List[float])->list[float]
    """ Return the y coordinate of the prism (floating base( walker as a float  
        I.E. using math.cos 
        TODO more documentation
    """
    y1_in = math.cos(state[2])
    y2_in = y1_in + math.cos(state[2] + state[3])
    y4_in = y2_in + math.cos(state[2] + state[3] + state[5])
    y5_in = y4_in + math.cos(state[2] + state[3] + state[5] + state[6])

    return -y5_in


def x_float(state):
    # type: (List[float])->list[float]
    """ 
        Return the x coordinate of the walker as a float  
        I.E. using math.sin
        TODO more documentation
    """
    x1_in = math.sin(state[0])
    x2_in = x1_in + math.sin(state[0] + state[1])
    x4_in = x2_in + math.sin(state[0] + state[1] + state[3])
    x5_in = x4_in + math.sin(state[0] + state[1] + state[3] + state[4])

    return -x5_in


def x_float_prism(state):
    # type: (List[float])->list[float]
    """ 
        Return the x coordinate of the walker as a float  
        I.E. using math.sin
        TODO more documentation
    """
    x1_in = math.sin(state[2])
    x2_in = x1_in + math.sin(state[2] + state[3])
    x4_in = x2_in + math.sin(state[2] + state[3] + state[5])
    x5_in = x4_in + math.sin(state[2] + state[3] + state[5] + state[6])

    return -x5_in


def y_taylor(state, degree):
    # type: (List[float])->list[float]
    """ 
        Return the x coordinate of the walker using a taylor series approximation
        I.E. using _taylor_sin (defined in this module)
        TODO more documentation
    """
    y1_in = _taylor_cos(state[0], degree)
    y2_in = y1_in + _taylor_cos(state[0] + state[1], degree)
    y4_in = y2_in + _taylor_cos(state[0] + state[1] + state[3], degree)
    y5_in = y4_in + _taylor_cos(state[0] + state[1] + state[3] + state[4], degree)

    return -y5_in


def x_taylor(state, degree=5):
    # type: (List[float])->list[float]
    """ 
        Return the y coordinate of the walker using a taylor series approximation
        I.E. using _taylor_cos (defined in this module)
        TODO more documentation
    """
    x1_in = _taylor_sin(state[0], degree)
    x2_in = x1_in + _taylor_sin(state[0] + state[1], degree)
    x4_in = x2_in + _taylor_sin(state[0] + state[1] + state[3], degree)
    x5_in = x4_in + _taylor_sin(state[0] + state[1] + state[3] + state[4], degree)

    return -x5_in


def y_taylor_prism(state, degree=5):
    # type: (List[float])->list[float]
    """ Return the y coordinate of the prism (floating base( walker as a float  
        I.E. using _taylor_cos 
        TODO more documentation
    """
    y1_in = _taylor_cos(state[2], degree)
    y2_in = y1_in + _taylor_cos(state[2] + state[3], degree)
    y4_in = y2_in + _taylor_cos(state[2] + state[3] + state[5], degree)
    y5_in = y4_in + _taylor_cos(state[2] + state[3] + state[5] + state[6], degree)

    return -y5_in


def x_taylor_prism(state, degree=5):
    # type: (List[float])->list[float]
    """ 
        Return the x coordinate of the walker as a float  
        I.E. using _taylor_sin
        TODO more documentation
    """
    x1_in = _taylor_sin(state[2], degree)
    x2_in = x1_in + _taylor_sin(state[2] + state[3], degree)
    x4_in = x2_in + _taylor_sin(state[2] + state[3] + state[5], degree)
    x5_in = x4_in + _taylor_sin(state[2] + state[3] + state[5] + state[6], degree)

    return -x5_in


def _taylor_sin(x, degree):
    sign = -1
    p = d = 1
    i = sinx = 0
    while p <= degree:
        d = (x ** p) / float(math.factorial(p))
        sinx += (sign ** i) * d
        i += 1
        p += 2
    return sinx


def _taylor_cos(x, degree):
    sign = 1
    d = 1
    p = 0
    i = cosx = 0
    while p <= degree:
        d = (x ** p) / float(math.factorial(p))
        cosx += (sign ** i) * d
        i += 1
        p += 2
    return cosx
