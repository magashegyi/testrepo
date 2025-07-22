# Quantum Toolkit
# Author: István Magashegyi
# Date: 2025 01 25
# Description: This module contains various functions and constants used in quantum mechanics simulations.

import numpy as np
import scipy.sparse as sparse
from . import potentials as pots
#from abc import ABC, abstractmethod # abstract base class
from typing import TypedDict, Optional
from atomic_units import hartree_atomic_base_units as hartree_atomic_units
#from utils.smooth_utils import smoothing1D
from abc import ABC, abstractmethod
import h5py

class Wavefunction:
    """Class representing a wavefunction in quantum mechanics.

    Attributes:
        grid (np.array): The spatial grid.
        value (np.array): The wavefunction values on the grid.
    """
    def __init__(self, grid: np.array, value: np.array):
        self.grid = grid
        self.value = value

    def to_hdf5_group(self, group):
        group.create_dataset("grid", data=self.grid)
        group.create_dataset("value", data=self.value)

    @classmethod
    def from_hdf5_group(cls, group):
        grid = group["grid"][()]
        value = group["value"][()]
        return cls(grid=grid, value=value)

class PlaneWave(Wavefunction):
    """Class representing a plane wave in quantum mechanics.
    
    This class extends Wavefunction to specifically handle plane waves
    with the ability to generate time-dependent or time-independent results.

    Attributes:
        grid (np.array): The spatial grid.
        value (np.array): The wavefunction values on the grid (initial spatial part).
        k (float): The wavenumber of the plane wave.
        omega (float, optional): The angular frequency of the plane wave. 
                                If None, only spatial dependence is available.
    """
    def __init__(self, k: float, omega: float = None, grid: np.array = None):
        """
        Initialize a PlaneWave object.

        Parameters:
        -----------
        k : float
            The wavenumber of the plane wave.
        omega : float, optional
            The angular frequency of the plane wave. If None, only spatial 
            dependence will be available.
        grid : np.array
            The spatial grid.
        """
        # Initialize the spatial part: exp(i*k*x)
        spatial_value = np.exp(1j * k * grid)
        super().__init__(grid, spatial_value)
        
        self.k = k
        self.omega = omega
    
    def __call__(self, times: np.array = None) -> 'Wavefunction':
        """
        Evaluate the plane wave at given time points.
        
        If times is None, returns the spatial part only (1D).
        If times is provided, returns the full time-dependent wave (2D).

        Parameters:
        -----------
        times : np.array, optional
            Array of time points. If None, returns only spatial dependence.

        Returns:
        --------
        Wavefunction
            A Wavefunction object containing either:
            - 1D array (nx,) if times is None: exp(i*k*x)
            - 2D array (nx, nt) if times is provided: exp(i*k*x - i*omega*t)
            
        Raises:
        -------
        ValueError
            If times is provided but omega was not set during initialization.
        """
        if times is None:
            # Return only spatial dependence
            return Wavefunction(grid=self.grid, value=self.value)
        
        # Time-dependent case
        if self.omega is None:
            raise ValueError("Cannot generate time-dependent wave: omega was not provided during initialization. "
                           "Create PlaneWave with omega parameter to enable time evolution.")
        
        nx = len(self.grid)
        nt = len(times)
        psi = np.zeros((nx, nt), dtype=np.complex128)
        
        # Calculate exp(i*k*x - i*omega*t) for all space-time points
        for i, t in enumerate(times):
            psi[:, i] = self.value * np.exp(-1j * self.omega * t)
        
        return Wavefunction(grid=self.grid, value=psi)
    
    def to_hdf5_group(self, group):
        """Save PlaneWave to HDF5 group."""
        super().to_hdf5_group(group)
        group.attrs["k"] = self.k
        if self.omega is not None:
            group.attrs["omega"] = self.omega
        else:
            group.attrs["omega"] = np.nan  # Use NaN to indicate no omega
        group.attrs["wave_type"] = "plane_wave"

    @classmethod
    def from_hdf5_group(cls, group):
        """Load PlaneWave from HDF5 group."""
        grid = group["grid"][()]
        k = group.attrs["k"]
        omega = group.attrs["omega"]
        # Handle NaN omega (no time dependence)
        if np.isnan(omega):
            omega = None
        return cls(k=k, omega=omega, grid=grid)

class Particle:
    """Class representing a particle in quantum mechanics.

    Attributes:
        charge (float): The charge of the particle.
        mass (float): The mass of the particle.
        angular_frequency (float): The angular frequency of the particle.
        wavenumber (float): The wavenumber of the particle.
        state (Wavefunction): The wavefunction representing the state of the particle.
    """
    def __init__(self, charge: float, mass: float, angular_frequency: float, wavenumber: float, state: Wavefunction):
        self.charge = charge
        self.mass = mass
        self.angular_frequency = angular_frequency
        self.wavenumber = wavenumber
        self.state = state

    def to_hdf5_group(self, group):
        group.attrs["charge"] = self.charge
        group.attrs["mass"] = self.mass
        group.attrs["angular_frequency"] = self.angular_frequency
        group.attrs["wavenumber"] = self.wavenumber
        state_grp = group.create_group("state")
        self.state.to_hdf5_group(state_grp)

    @classmethod
    def from_hdf5_group(cls, group):
        charge = group.attrs["charge"]
        mass = group.attrs["mass"]
        angular_frequency = group.attrs["angular_frequency"]
        wavenumber = group.attrs["wavenumber"]
        state = Wavefunction.from_hdf5_group(group["state"])
        return cls(
            charge=charge,
            mass=mass,
            angular_frequency=angular_frequency,
            wavenumber=wavenumber,
            state=state
        )

def zero_wave_function(grid: np.array) -> Wavefunction:
    """
    Generate a zero wavefunction.

    Parameters:
    grid (np.array): The spatial grid.

    Returns:
    Wavefunction: The zero wavefunction.
    """
    return Wavefunction(grid=grid, value=np.zeros_like(grid, dtype=np.complex128))

def plane_wave(k: float, grid: np.array, omega: float = None) -> PlaneWave:
    """
    Create a PlaneWave object with optional time-evolution capabilities.
    
    This function creates a PlaneWave that can generate time-dependent
    results only if omega is provided, otherwise only spatial dependence.

    Parameters:
    -----------
    k : float
        The wavenumber of the plane wave.
    grid : np.array
        The spatial grid.
    omega : float, optional
        The angular frequency of the plane wave. If not provided,
        only spatial wavefunction will be available.

    Returns:
    --------
    PlaneWave
        A PlaneWave object that can be called with optional time array
        (only if omega was provided).
        
    Examples:
    ---------
    >>> # Only spatial dependence
    >>> pw_spatial = plane_wave(k=1.0, grid=np.linspace(0, 10, 100))
    >>> spatial_only = pw_spatial()  # Returns 1D spatial wavefunction
    >>> # pw_spatial(times) would raise ValueError
    
    >>> # With time dependence
    >>> pw_time = plane_wave(k=1.0, grid=np.linspace(0, 10, 100), omega=2.0)
    >>> spatial_only = pw_time()  # Returns 1D spatial wavefunction
    >>> times = np.linspace(0, 5, 50)
    >>> time_dependent = pw_time(times)  # Returns 2D space-time wavefunction
    """
    return PlaneWave(k=k, omega=omega, grid=grid)

def gaussian_wavepacket(x0: float, p0: float, sigma: float, grid: np.array) -> Wavefunction:
    """
    Generate a Gaussian wave packet.

    Parameters:
    x0 (float): The initial position of the wave packet.
    p0 (float): The initial momentum of the wave packet.
    sigma (float): The width of the wave packet.
    grid (np.array): The spatial grid.

    Returns:
    Wavefunction: The Gaussian wave packet.
    """
    return Wavefunction(grid=grid, value=(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-((grid-x0)**2)/(2*sigma**2))*np.exp(1j*p0*grid))

def mask_wavefunction(wavefunction: Wavefunction, mask: np.ndarray) -> Wavefunction:
    """
    Apply a mask to the wavefunction to store only certain points and their corresponding values.

    Parameters:
    wavefunction (Wavefunction): The original wavefunction.
    mask (np.ndarray): The mask that determines which points to keep.

    Returns:
    Wavefunction: The masked wavefunction.
    """
    masked_grid = wavefunction.grid[mask]
    masked_value = wavefunction.value[mask]
    return Wavefunction(grid=masked_grid, value=masked_value)

def probability_density(state: Wavefunction):
    return (np.conj(state.value)*state.value).real

# def probability_current(state,dx):  
#     return np.imag(np.conj(state) * np.gradient(state,dx,axis=0))

def probability_current(state: Wavefunction):
    grid=state.grid
    dx=grid[1]-grid[0]
    return np.imag(np.conj(state.value) * np.gradient(state.value,dx,axis=0))

def probability_current_at(state: Wavefunction, idx: int):
    """
    Valószínűségi áram számítása egy adott indexű pontban.

    Paraméterek:
    state (Wavefunction): Hullámfüggvény.
    idx (int): A vizsgált pont indexe.

    Visszatérési érték:
    float: Valószínűségi áram az adott pontban.
    """
    grid = state.grid
    dx = grid[1] - grid[0]
    grad = np.gradient(state.value, dx, axis=0)
    return np.imag(np.conj(state.value[idx]) * grad[idx])

# class ConstT:

#     def __init__(self,dx: float,\
#                   hbar: float = hartree_atomic_units["action"],\
#                   m: float = hartree_atomic_units["mass"]):
#         self.__t=(hbar**2)/(2*m*(dx**2))

#     @property
#     def __call__(self) -> float:
#         return self.__t

def const_t(dx: float,hbar: float=hartree_atomic_units.hb,m: float=hartree_atomic_units.me) -> float:
    return (hbar**2)/(2*m*(dx**2))

# def const_t(hb: float,p: Particle):
#     dx=p.state.grid[1]-p.state.grid[0]
#     hbar=hb
#     mass=p.mass
#     return const_t(dx,hbar,mass)

# def const_t(au: AtomicUnits,p: Particle):
#     hbar=au.action
#     return const_t(hbar,p)

class BaseDispersion(ABC):
    """
    Abstract base class for dispersion relations.

    Attributes:
        params (dict): Generalized parameter dictionary for calculations.
    """
    def __init__(self, **params):
        """
        Initialize the BaseDispersion class with arbitrary parameters.

        Parameters:
        -----------
        **params : dict
            Arbitrary keyword arguments for initialization.
        """
        self.params = params

    @abstractmethod
    def wavenumber(self, energy: float) -> float:
        """
        Convert energy to wave number (k).

        Parameters:
        -----------
        energy : float
            The energy value.

        Returns:
        --------
        float
            The corresponding wave number (k).
        """

        if "wavenumber" in self.params:
            return self.params["wavenumber"](energy, **self.params)
        raise NotImplementedError("wavenumber function is not defined.")

    @abstractmethod
    def energy(self, wavenumber: float) -> float:
        """Convert wave number to energy."""
        pass

    @abstractmethod
    def angular_frequency(self, wavenumber: float) -> float:
        """Convert wave number to angular frequency (omega)."""
        pass


class CosineDispersion(BaseDispersion):
    """
    Dispersion relation based on a cosine function.
    """
    def __init__(self, step_size: float, hbar: float, mass: float):
        super().__init__(step_size=step_size, hbar=hbar, mass=mass)
        self._t = (hbar**2) / (2 * mass * (step_size**2))

    def wavenumber(self, energy: float) -> float:
        step_size = self.params["step_size"]
        return (1 / step_size) * np.arccos(1 - (0.5 * energy / self._t))

    def energy(self, wavenumber: float) -> float:
        step_size = self.params["step_size"]
        return 2 * self._t * (1 - np.cos(wavenumber * step_size))

    def angular_frequency(self, wavenumber: float) -> float:
        hbar = self.params["hbar"]
        return self.energy(wavenumber) / hbar


class QuadraticDispersion(BaseDispersion):
    """
    Dispersion relation based on a quadratic function.
    """
    def __init__(self, hbar: float, mass: float):
        super().__init__(hbar=hbar, mass=mass)

    def wavenumber(self, energy: float) -> float:
        hbar = self.params["hbar"]
        mass = self.params["mass"]
        #print(f"energy: {energy}, hbar: {hbar}, mass: {mass}")
        print(f"wavenumber: {np.sqrt(2 * mass * energy) / hbar}")
        return np.sqrt(2 * mass * energy) / hbar

    def energy(self, wavenumber: float) -> float:
        hbar = self.params["hbar"]
        mass = self.params["mass"]
        return (hbar**2 * wavenumber**2) / (2 * mass)

    def angular_frequency(self, wavenumber: float) -> float:
        hbar = self.params["hbar"]
        return self.energy(wavenumber) / hbar

# """Class for calculating dispersion relations.

# Attributes:
#     unified_step_size (float): The step size for the calculations.
#     action (float): The action unit.
#     mass (float): The mass unit.
# """
# class Dispersion:

#     def __init__(self,unified_step_size: float,\
#                   hbar: float = hartree_atomic_units.hb,\
#                   mass: float = hartree_atomic_units.me):

#         self.__hbar=hbar
#         self.__step_size=unified_step_size
#         self.__t=const_t(unified_step_size,hbar,mass)

#     """Convert energy to wave number (k).

#     Args:
#         energy (float): The energy value.

#     Returns:
#         float: The corresponding wave number (k).
#     """
#     def energy2k(self,energy: float) -> float:
#         k=(1/self.__step_size)*np.arccos(1-(0.5*energy/self.__t))
#         return k
    
#     """Convert wave number (k) to energy.

#     Args:
#         k (float): The wave number.

#     Returns:
#         float: The corresponding energy.
#     """
#     def k2energy(self,k: float) -> float:
#         E=2*self.__t*(1-np.cos(k*self.__step_size))
#         return E
    
#     """Convert wave number (k) to angular frequency (omega).

#     Args:
#         k (float): The wave number.

#     Returns:
#         float: The corresponding angular frequency.
#     """
#     def k2omega(self,k: float) -> float:
#         return self.k2energy(k)/self.__hbar


# def inverse_dispersion(k,dx=1,hbar=hartree_atomic_units["action"],m=hartree_atomic_units["mass"]):
#     t=const_t(dx,hbar,m)
#     E=2*t*(1-np.cos(k*dx))
#     return E

# def dispersion(energy,dx=1, hbar=hartree_atomic_units["action"], m=hartree_atomic_units["mass"]):
#     t=const_t(dx,hbar,m)
#     k=(1/dx)*np.arccos(1-(0.5*energy/t))
#     return k
    
"""Class representing the Hamiltonian operator in quantum mechanics.

Attributes:
    uxgrid (np.array): The spatial grid.
    scalarpot (function): The scalar potential function.
    vectorpot (function): The vector potential function.
    me (float): The mass of the particle.
    hbar (float): The reduced Planck constant.
    charge (float): The charge of the particle.
"""
class HamiltonOperator:
    def __init__(self, uxgrid=np.linspace(0, 1, 201), \
                 scalarpot=pots.ZeroPotential, \
                 vectorpot=pots.ZeroPotential, \
                 me=1.0, hbar=1.0, charge=1):
        
        self.__me=me
        self.__hbar=hbar
        self.__xgrid=uxgrid
        self.__dx=uxgrid[1]-uxgrid[0]
        self.__spot=scalarpot
        self.__vpot=vectorpot
        self.__size=len(self.__xgrid)
        self.__charge=charge
        self.__c1 = 1j*self.__charge*self.__dx/self.__hbar
        self.__c2 = -(self.__hbar**2)/(2*self.__me)

        # Create the matrix containing central 
        # differences. It it used to
        # approximate the second derivative.
        D2 = (1/(self.__dx**2))*np.ones((3, self.__size), dtype=np.complex128)
        D2[1] = -2*D2[1]

        # Kinetic energy operator
        self.__Kinetic = self.__c2*D2

        self.__Vcap=np.zeros((3, self.__size), dtype=np.complex128)
        self.__ham_diags=np.zeros((3, self.__size), dtype=np.complex128)

    def __str__(self):
        return f"Hamiltonian"
    
    def __call__(self,time):

        # load kinetic part
        self.__ham_diags = np.copy(self.__Kinetic) #+ np.copy(self.__Vcap)

        # add scalar potential
        self.__ham_diags[1] += self.__spot(time)#self.__charge*self.__spot(time)

        # vector potential
        self.__ham_diags[2] = self.__ham_diags[2]*np.exp(-self.__c1*self.__vpot(time))
        self.__ham_diags[0] = self.__ham_diags[0]*np.exp(self.__c1*self.__vpot(time))

        # load complex absorbing potential
        self.__ham_diags += np.copy(self.__Vcap)

        H = sparse.spdiags(self.__ham_diags, [-1,0,1], self.__size, self.__size)

        return H
    
    """Set the scalar potential.

    Args:
        spot (function): The scalar potential function.
    """
    def scalar_potential(self, spot):
        self.__spot=spot

    """Set the vector potential.

    Args:
        vpot (function): The vector potential function.
    """
    def vector_potential(self, vpot):
        self.__vpot=vpot

    """Set the complex absorbing potential.

    Args:
        vcap (np.array): The complex absorbing potential array.
    """
    def vcap(self, vcap):
        self.__Vcap=vcap

    #diags accessor
    @property
    def diags(self):
        return self.__ham_diags

    #hbar accessor
    @property
    def hbar(self):
        return self.__hbar

    #mass accessor
    @property
    def mass(self):
        return self.__me
    
    #mass accessor
    @property
    def charge(self):
        return self.__charge
    
    #mass accessor
    @property
    def stepsize(self):
        return self.__dx
    
    #kinetic energy operator
    @property
    def kineticenergyoperator(self):
        return self.__Kinetic
    
    #grid accessor
    @property
    def grid(self):
        return self.__xgrid
    
def vcap_generator(uxgrid=np.linspace(0, 1, 201),\
                mass=hartree_atomic_units.me,\
                hbar=hartree_atomic_units.hb,\
                param_x0 = None, param_lambda0 = 0.05,\
                param_theta0 = 0.4 ):

    dx = uxgrid[1]-uxgrid[0]
    xnum = len(uxgrid)
    lamb = param_lambda0

    if param_x0 == None: 
        x0=0.75*uxgrid[-1]#300
    else:
        x0=param_x0

    theta0=param_theta0
    a=1+0.5*(np.tanh(lamb*(uxgrid-x0))-np.tanh(lamb*(uxgrid+x0)))
    theta=a*theta0
    
    g=((np.exp(1j*theta)-1)/(np.exp(1j*theta0)-1)) + 1j*uxgrid*(np.exp(1j*theta)/(np.exp(1j*theta0)-1))*theta0*lamb*0.5*( (1/np.cosh(lamb*(uxgrid-x0)))**2 - (1/np.cosh(lamb*(uxgrid+x0)))**2)
    f=1+(np.exp(1j*theta0)-1)*g
    dfdx=np.gradient(f,dx)
    d2fdx2=np.gradient(dfdx,dx)
    
    V0 = ( (hbar**2) / (4*mass*f**3) )*d2fdx2 - ( (5*hbar**2) / (8*mass*f**4) )*(dfdx**2)
    V1 = ( (hbar**2) / (mass*f**3) )*dfdx
    V2 = ( (hbar**2) / (2*mass) )*( 1 - (1/(f**2)) )

    # Create the matrix containing central 
    # differences. It it used to
    # approximate the first derivative.
    D1 = (1/(2*dx))*np.ones((3, xnum), dtype=np.complex128)
    D1[0] = -1*D1[0]
    D1[1] = 0*D1[1]

    # Create the matrix containing central 
    # differences. It it used to
    # approximate the second derivative.
    D2 = (1/(dx**2))*np.ones((3, xnum), dtype=np.complex128)
    D2[1] = -2*D2[1]

    Vcap = V0 + V1*D1 + V2*D2
    
    return Vcap

#Stationary state time evolution
def stationary_time_evolution(omega: float, times: np.array, eigenstate: Wavefunction) -> Wavefunction:
    """
    Compute the time evolution of an eigenstate under a given frequency.

    Parameters:
    -----------
    omega : float
        The angular frequency of the time evolution.
    times : np.array
        Array of time points at which to evaluate the time evolution.
    eigenstate : Wavefunction
        The initial eigenstate represented as a Wavefunction object, containing 'grid' and 'value'.

    Returns:
    --------
    Wavefunction
        A Wavefunction object containing the grid and the time-evolved values at each time point.
    """

    nx=len(eigenstate.grid)
    nt=len(times)
    psi = np.zeros((nx,nt),dtype=np.complex128)

    eigenstate_value = eigenstate.value
    psi[:,0] = eigenstate_value
    for i, t in enumerate(times[1:], start=1):
        psi[:,i] = np.exp(-1j*omega*t)*eigenstate_value

    return Wavefunction(grid=eigenstate.grid, value=psi)



"""Class for solving the time-dependent Schrödinger equation using the Crank-Nicholson method.

Attributes:
    inistate (np.array): The initial state wavefunction.
    ham (HamiltonOperator): The Hamiltonian operator.
    dt (float): The time step size.
    t_start (float): The initial time.
"""
class CntdSes:
    def __init__(self, inistate, ham, dt=0.01, t_start=0.0):
        self.__state=inistate
        self.__ham=ham

        self.__dt=dt
        self.__t=t_start

        self.__size=len(ham.grid)
        self.__I=sparse.identity(self.__size) # Identity Matrix
        self.__alpha=1j*dt/(2*ham.hbar)

    "default boundary is Dirichlet"
    def boundary(self,time):
        return np.zeros_like(self.__state)

    """
        Btdt = boundary matrix at t=t_i+dt
        Bt   = boundary matrix at t=t_i
        alpha = i*dt/(2*hbar)
        We want to solve the system of linear equations int the form of:
        (I + alpha*H(t+dt)) psi_new = (I - alpha*H(t)) psi_old - alpha*(B(t+dt)+B(t))
                        Mleft psi_new = Mright psi_old - alpha*(B(t+dt)+B(t))
    """

    """Set the boundary condition.

    Args:
        time (float): The current time.

    Returns:
        np.array: The boundary condition array.
    """
    def step_one(self):
        H=self.__ham
        t=self.__t
        dt=self.__dt
        psi_old=self.__state
        I=self.__I
        alpha=self.__alpha

        B=self.boundary
        #print(B(t),B(t+dt))

        Mleft = I + alpha*H(t+dt)
        Mright = I - alpha*H(t)

        """
        Solve the system of linear equations in the following form:
        (I + alpha*H) psi_new = (I - alpha*H) psi_old - alpha*(B(t+dt) + B(t))
                            A psi_new = b
        """

        # Set the elements of the equation
        A = sparse.csr_matrix(Mleft)
        b = Mright @ psi_old - alpha*(B(t+dt) + B(t))

        #Solve the system of linear equations
        psi_new = sparse.linalg.spsolve(A,b)

        self.__t=t+dt
        self.__state=psi_new

        return self.__state
    
    """Perform multiple time steps of the Crank-Nicholson method.

    Args:
        n (int): The number of time steps to perform.
        observer (Optional[Callable]): Opcionális függvény, amit minden lépés után meghívunk az aktuális állapottal.

    Returns:
        np.array: The updated state wavefunction.
    """
    def step_n(self, n=2, observer=None):
        if observer is None:
            for i in range(n):
                self.step_one()
        else:
            for i in range(n):
                self.step_one()
                observer(self.__state, i)
        return self.__state

    def __str__(self):
        return f"Crank-Nicholson methode based Time Dependent Schrödinger Equation Solver {self.__dt} {self.__ts}"
    
"""Class for generating split boundary conditions for time evolution calculations.

Attributes:
    hham (HamiltonOperator): The Hamiltonian operator with vector potential.
    h0ham (HamiltonOperator): The Hamiltonian operator without vector potential.
    psi0 (Wavefunction): The initial wavefunction.
    omega (float): The angular frequency.
"""
class SplitBoundary:
    def __init__(self, hham:HamiltonOperator, h0ham:HamiltonOperator,\
                 psi0: Wavefunction,omega:float):
        self.__hham=hham
        self.__h0ham=h0ham
        self.__psi0_initial=psi0.value
        self.__omega=omega

    def ham1(self,time):
        return self.__hham(time)-self.__h0ham(time)

    """Calculate the boundary condition at a given time.

    Args:
        time (float): The current time.

    Returns:
        np.array: The boundary condition array.
    """
    def __call__(self,time):
        h1=self.__hham(time)-self.__h0ham(time)

        psi0_at_time=np.exp(-1j*self.__omega*time)*self.__psi0_initial
        B = h1 @ psi0_at_time
        return B


class SplitTimeEvolutionCalculator:
    """
    Class for calculating the time evolution of a wavefunction using split operator methods.
    This simplified version does not store internal memory or support masking.
    Results are provided via an observer function at each time step.

    Attributes:
        psi0_omega (float): The angular frequency of the initial wavefunction.
        psi0_initial (Wavefunction): The initial wavefunction.
        scalarpot (function): The scalar potential function.
        vectorpot (function): The vector potential function.
        me (float): The mass of the particle.
        hbar (float): The reduced Planck constant.
        charge (float): The charge of the particle.
        dt (float): The time step size.
        t_start (float): The initial time.
        t_stop (float): The final time.
        vcap (Optional[np.ndarray]): The complex absorbing potential array.
    """
    def __init__(self, psi0_omega: float, psi0_initial: Wavefunction,
                 psi1_initial: Wavefunction = None,
                 scalarpot=pots.ZeroPotential, vectorpot=pots.ZeroPotential,
                 me = None,#=hartree_atomic_units.me,
                 hbar = None,#=hartree_atomic_units.hb,
                 charge = None,#=hartree_atomic_units.e0,
                 dt=0.01, t_start=0.0, t_stop=1.0, vcap=None):
        """
        Initialize the SplitTimeEvolutionCalculator.

        Parameters:
        -----------
        psi0_omega : float
            The angular frequency of the initial wavefunction.
        psi0_initial : Wavefunction
            The initial wavefunction.
        psi1_initial : Wavefunction, optional
            The initial state for the driven part (default: zero).
        scalarpot : function
            The scalar potential function.
        vectorpot : function
            The vector potential function.
        me : float
            The mass of the particle.
        hbar : float
            The reduced Planck constant.
        charge : float
            The charge of the particle.
        dt : float
            The time step size.
        t_start : float
            The initial time.
        t_stop : float
            The final time.
        vcap : Optional[np.ndarray], optional
            The complex absorbing potential array.
        """
        self.__psi0_omega = psi0_omega
        self.__psi0_initial = psi0_initial
        self.__utgrid = np.arange(t_start, t_stop, dt)

        if not hasattr(scalarpot, 'grid'):
            raise AttributeError("The scalar potential object must have a 'grid' attribute.")

        self.__uxgrid = scalarpot.grid
        zeropot = pots.ZeroPotential(self.__uxgrid)

        hham2h1 = HamiltonOperator(self.__uxgrid, scalarpot=scalarpot,
                                   vectorpot=vectorpot,
                                   me=me, hbar=hbar, charge=charge)
        h0ham2h1 = HamiltonOperator(self.__uxgrid, scalarpot=scalarpot,
                                    vectorpot=zeropot,
                                    me=me, hbar=hbar, charge=charge)
        hham = HamiltonOperator(self.__uxgrid, scalarpot=scalarpot,
                                vectorpot=vectorpot,
                                me=me, hbar=hbar, charge=charge)

        if vcap is not None:
            hham.vcap(vcap)

        self.__nx = len(self.__uxgrid)
        self.__nt = len(self.__utgrid)

        if psi1_initial is not None:
            self.__psi1_initial = np.array(psi1_initial.value, dtype=np.complex128)
        else:
            self.__psi1_initial = np.zeros(self.__nx, dtype=np.complex128)

        self.__solver = CntdSes(self.__psi1_initial, hham, dt=dt)
        sb = SplitBoundary(hham2h1, h0ham2h1, psi0_initial, psi0_omega)
        self.__solver.boundary = sb

    def run(self, observer):
        """
        Run the time evolution calculation.

        The observer function will be called at each time step with the following arguments:
            observer(psi0, psi1, psi, uxgrid, current_time, time_index, utgrid)
        where:
            psi0: np.ndarray, the stationary part at the current time
            psi1: np.ndarray, the driven part at the current time
            psi: np.ndarray, the total state at the current time (psi0 + psi1)
            uxgrid: np.ndarray, the spatial grid
            current_time: float, the current time
            time_index: int, the current time index
            utgrid: np.ndarray, the full time grid
        """
        psi0_evol = stationary_time_evolution(self.__psi0_omega,
                                              self.__utgrid, self.__psi0_initial)
        psi1 = self.__psi1_initial
        observer(psi0_evol.value[:, 0], psi1, psi0_evol.value[:, 0] + psi1, self.__uxgrid, self.__utgrid[0], 0, self.__utgrid)
        for i in range(1, self.__nt):
            psi1 = self.__solver.step_one()
            psi0 = psi0_evol.value[:, i]
            psi = psi0 + psi1
            observer(psi0, psi1, psi, self.__uxgrid, self.__utgrid[i], i, self.__utgrid)
        #     if i % save_step == 0:
        #         self.__psi1_time_evolution[:,i]=self.__solver.step_one()
        #     else:
        #         self.__solver.step_one()

        # if self.__mask is None:
        #     self.__psi0_time_evolution = stationary_time_evolution(self.__psi0_omega,\
        #                                     self.__utgrid, self.__psi0_initial)

        #     if self.__psi1_initial is not None:
        #         #print(self.__psi1_time_evolution.shape[0],self.__psi1_initial.shape[0])
        #         self.__psi1_time_evolution[:,0]=self.__psi1_initial
        #     else:
        #         self.__psi1_time_evolution[:,0]=np.zeros(self.__nx,dtype=np.complex128)

        #     for i in range(1,self.__nt):
        #         self.__psi1_time_evolution[:,i]=self.__solver.step_one()

        # else:
        #     self.__psi0_time_evolution = stationary_time_evolution(self.__psi0_omega,\
        #                                     self.__utgrid, self.__psi0_initial, self.__mask)

        #     if self.__psi1_initial is not None:
        #         self.__psi1_time_evolution[:,0]=self.__psi1_initial[self.__mask]
        #     else:
        #         self.__psi1_time_evolution[:,0]=np.zeros(self.__masked_nx,dtype=np.complex128)
        #     for i in range(1,self.__nt):
        #         self.__psi1_time_evolution[:,i]=self.__solver.step_one()[self.__mask]

    @property
    def psi0_time_evolution(self):
        return self.__psi0_time_evolution
    
    @property
    def psi1_time_evolution(self):
        return Wavefunction(grid=self.__uxgrid,value=self.__psi1_time_evolution)
    
    @property
    def psi_time_evolution(self) -> Wavefunction:
        psi = self.__psi1_time_evolution + self.__psi0_time_evolution.value
        return Wavefunction(grid=self.__uxgrid,value=psi)
    
    @property
    def boundary(self):
        return self.__solver.boundary


class SimpleTimeEvolutionCalculator:
    """
    Class for calculating the time evolution of a wavefunction using a single Hamiltonian operator.
    This simplified version does not use split operator methods and evolves the entire state 
    with a single Hamiltonian operator. Results are provided via an observer function at each time step.

    Attributes:
        initial_state (Wavefunction): The initial wavefunction.
        scalarpot (function): The scalar potential function.
        vectorpot (function): The vector potential function.
        me (float): The mass of the particle.
        hbar (float): The reduced Planck constant.
        charge (float): The charge of the particle.
        dt (float): The time step size.
        t_start (float): The initial time.
        t_stop (float): The final time.
        vcap (Optional[np.ndarray]): The complex absorbing potential array.
    """
    def __init__(self, initial_state: Wavefunction,
                 scalarpot=pots.ZeroPotential, vectorpot=pots.ZeroPotential,
                 me=hartree_atomic_units.me,
                 hbar=hartree_atomic_units.hb,
                 charge=hartree_atomic_units.e0,
                 dt=0.01, t_start=0.0, t_stop=1.0, vcap=None):
        """
        Initialize the SimpleTimeEvolutionCalculator.

        Parameters:
        -----------
        initial_state : Wavefunction
            The initial wavefunction.
        scalarpot : function
            The scalar potential function.
        vectorpot : function
            The vector potential function.
        me : float
            The mass of the particle.
        hbar : float
            The reduced Planck constant.
        charge : float
            The charge of the particle.
        dt : float
            The time step size.
        t_start : float
            The initial time.
        t_stop : float
            The final time.
        vcap : Optional[np.ndarray], optional
            The complex absorbing potential array.
        """
        self.__initial_state = initial_state
        self.__utgrid = np.arange(t_start, t_stop, dt)

        if not hasattr(scalarpot, 'grid'):
            raise AttributeError("The scalar potential object must have a 'grid' attribute.")

        self.__uxgrid = scalarpot.grid

        # Create the main Hamiltonian operator
        hham = HamiltonOperator(self.__uxgrid, scalarpot=scalarpot,
                                vectorpot=vectorpot,
                                me=me, hbar=hbar, charge=charge)

        if vcap is None:
            self.v_cap = vcap_generator(self.__uxgrid)
        else:
            self.v_cap = vcap
        hham.vcap(self.v_cap)

        self.__nx = len(self.__uxgrid)
        self.__nt = len(self.__utgrid)

        # Initialize the solver with the initial state
        initial_state_array = np.array(initial_state.value, dtype=np.complex128)
        self.__solver = CntdSes(initial_state_array, hham, dt=dt, t_start=t_start)

    def run(self, observer):
        """
        Run the time evolution calculation.

        The observer function will be called at each time step with the following arguments:
            observer(psi, uxgrid, current_time, time_index, utgrid)
        where:
            psi: np.ndarray, the state at the current time
            uxgrid: np.ndarray, the spatial grid
            current_time: float, the current time
            time_index: int, the current time index
            utgrid: np.ndarray, the full time grid
        """
        # Call observer for initial state
        psi = self.__solver._CntdSes__state  # Access private attribute
        observer(psi, self.__uxgrid, self.__utgrid[0], 0, self.__utgrid)
        
        # Time evolution loop
        for i in range(1, self.__nt):
            psi = self.__solver.step_one()
            observer(psi, self.__uxgrid, self.__utgrid[i], i, self.__utgrid)

    @property
    def grid(self):
        """Get the spatial grid."""
        return self.__uxgrid
    
    @property
    def time_grid(self):
        """Get the time grid."""
        return self.__utgrid
    
    @property
    def initial_state(self):
        """Get the initial state."""
        return self.__initial_state


# class BaseDispersion(ABC):
#     """
#     Abstract base class for dispersion relations.

#     Attributes:
#         step_size (float): The step size for the calculations.
#         hbar (float): The reduced Planck constant.
#         mass (float): The mass of the particle.
#     """
#     def __init__(self, step_size: float, hbar: float, mass: float):
#         self._step_size = step_size
#         self._hbar = hbar
#         self._mass = mass

#     @abstractmethod
#     def energy2k(self, energy: float) -> float:
#         """Convert energy to wave number (k)."""
#         pass

#     @abstractmethod
#     def k2energy(self, k: float) -> float:
#         """Convert wave number (k) to energy."""
#         pass

#     @abstractmethod
#     def k2omega(self, k: float) -> float:
#         """Convert wave number (k) to angular frequency (omega)."""
#         pass

# class CosineDispersion(BaseDispersion):
#     """
#     Dispersion relation based on a cosine function.
#     """
#     def __init__(self, step_size: float, hbar: float, mass: float):
#         super().__init__(step_size, hbar, mass)
#         self._t = (hbar**2) / (2 * mass * (step_size**2))

#     def energy2k(self, energy: float) -> float:
#         return (1 / self._step_size) * np.arccos(1 - (0.5 * energy / self._t))

#     def k2energy(self, k: float) -> float:
#         return 2 * self._t * (1 - np.cos(k * self._step_size))

#     def k2omega(self, k: float) -> float:
#         return self.k2energy(k) / self._hbar

# class QuadraticDispersion(BaseDispersion):
#     """
#     Dispersion relation based on a quadratic function.
#     """
#     def energy2k(self, energy: float) -> float:
#         return np.sqrt(2 * self._mass * energy) / self._hbar

#     def k2energy(self, k: float) -> float:
#         return (self._hbar**2 * k**2) / (2 * self._mass)

#     def k2omega(self, k: float) -> float:
#         return self.k2energy(k) / self._hbar
