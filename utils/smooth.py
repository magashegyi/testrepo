import numpy as np

def smoothing1D_psi(x):
    """
    Generates the smoothing psi function for positive x values.
    """
    x_half = np.where(x > 0, x, 1)
    return np.where(x > 0, np.exp(-1 / x_half), 0)


def smoothing1D_phi(x, psi=None):
    """
    Generates the smoothing phi function based on psi(x).
    """
    if psi is None:
        psi = smoothing1D_psi

    psi_x = psi(x)
    psi_1_x = psi(1 - x)

    return np.where((x > 0) & (x < 1), psi_x / (psi_x + psi_1_x), 0) + np.where(x >= 1, 1, 0)


def smoothing1D(x, f, g, a=0, b=1, psi=None):
    """
    Smoothly transitions between two functions f and g over the range [a, b].
    """
    if b < a:
        raise ValueError("Parameter 'b' must be greater than 'a'.")

    lx = (x - a) / (b - a)
    phi = smoothing1D_phi(lx, psi)

    return (1 - phi) * f(x) + phi * g(x)

def smooth_jumper(t, jumps, default_beta=10):
    """
    Generates a smoothed step function f(t) using hyperbolic tangent.

    Parameters:
        t (numpy.ndarray): The time range array.
        jumps (list of tuples): Each tuple contains (location, height) or (location, height, beta).
        default_beta (float): Default sharpness parameter if beta is not provided.

    Returns:
        numpy.ndarray: The generated function values f(t).
    """
    f = np.zeros_like(t)
    for jump in jumps:
        if len(jump) == 2:  # If beta is not provided
            location, height = jump
            beta = default_beta
        elif len(jump) == 3:  # If beta is provided
            location, height, beta = jump
        else:
            raise ValueError("Each jump must be a tuple with 2 or 3 values (location, height, [beta]).")
        
        f += height * 0.5 * (1 + np.tanh(beta * (t - location)))
    return f
