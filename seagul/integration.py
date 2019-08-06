def rk4(derivs, a, t0, dt, s0):
    """
    Single step of an RK4 solver, designed for control applications, so it passed an action to your
    derivs fcn

    Attributes:
        derivs: the function you are trying to integrate, should have signature:
        function(t,s,a) -> ds/dt

        a: action, should belong to the action space of your environment

        t0: float, initial time, often you can just set this to zero if all that matters for your
        derivs is the state and dt

        dt: how big of a timestep to integrate

        s0: initial state of your system, should belong to the envs obvervation space

    Returns:
        s[n+1]: I.E. the state of your system after integrating with action a for dt seconds

    Example:
        derivs = lambda t,q,a: (q+a)**2
        a =  1.0
        t0 = 0
        dt = .1
        s0 = 5
        s1 = rk4(derivs, a, t0, dt, s0)

    """

    k1 = dt * derivs(t0, s0, a)
    k2 = dt * derivs(t0 + dt / 2, s0 + k1 / 2, a)
    k3 = dt * derivs(t0 + dt / 2, s0 + k2 / 2, a)
    k4 = dt * derivs(t0 + dt, s0 + k3, a)

    return s0 + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def euler(derivs, a, t0, dt, s0):
    """
    Single step of an euler integtation, exactly the same parameters and usage as rk4 above
    """
    return s0 + dt * derivs(t0 + dt, s0, a)
