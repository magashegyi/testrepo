from scipy.optimize import minimize
import atomic_units as au
import numpy as np

def ct_calculator(en, dpl):
    """
    Calculate the ct value based on energy, desired points in lambda, and grid step size.

    Parameters:
    energy (float): Energy in eV.
    desired_points_in_lamda (int): Desired points in lambda.
    dx (float): Grid step size in the unit system.

    Returns:
    float: Calculated ct value.
    """
    return en / (2 * (1 - np.cos(2 * np.pi / dpl)))

def optimize_atomic_units(dx, npx, emax, desired_points_in_lambda, desired_width):
    """
    Optimize atomic unit parameters to minimize errors in prefactor and width.

    Parameters:
    dx (float): Grid step size in the unit system.
    emax (float): Maximum electron energy in eV.
    desired_width (float): Desired width in nm.

    Returns:
    dict: Optimization results including optimized parameters and objective value.
    """
    desired_ct = ct_calculator(emax, desired_points_in_lambda)

    def objective_function(x):
        hb, m0, k0, xe = x
        usys = au.AtomicUnitSystem(xh=hb, xe=xe, xm=m0, xk=k0)

        desired_width_in_nm = desired_width * usys.ureg("nm")
        width_in_nm = npx * dx * usys.length_unit.to("nm")
        delta_width = (abs(width_in_nm - desired_width_in_nm) / desired_width_in_nm).magnitude

        ct = (hb**2) / (2 * m0 * dx**2)
        delta_ct = 1e4*(ct - desired_ct) / desired_ct

        points_in_lambda = 2.0*np.pi/(0.5*np.pi-1 + (emax/(2.0*ct)))
        delta_nl = 1e0*(points_in_lambda - desired_points_in_lambda)/desired_points_in_lambda

        return delta_nl**2 + delta_width**2

    result = minimize(objective_function, [1, 1, 1, 1], bounds=((1, 10), (1, 10), (1, 10), (1, 10)), method="L-BFGS-B")

    print(f"Optimized parameters: {result.x[0]:.1f}, {result.x[1]:.1f}, {result.x[2]:.1f}, {result.x[3]:.1f}")
    print(f"Objective function value: {result.fun:.4f}")

    result = minimize(objective_function, [result.x[0], result.x[1], result.x[2], result.x[3]], bounds=((1, 10), (1, 1000), (1, 1000), (1, 1000)), method="L-BFGS-B")

    return {
        "optimized_parameters": result.x,
        "objective_value": result.fun
    }
