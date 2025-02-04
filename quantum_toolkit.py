# Quantum Toolkit
# Author: István Magashegyi
# Date: 2025 01 25
# Description: This module contains various functions and constants used in quantum mechanics simulations.

import numpy as np
import scipy.sparse as sparse
#from abc import ABC, abstractmethod # abstract base class
from typing import TypedDict, Optional
import potentials as pots

"""Class representing atomic units used in quantum mechanics.

Attributes:
    action (float): The action unit.
    charge (float): The charge unit.
    mass (float): The mass unit.
    permittivity (float): The permittivity unit.
"""
class AtomicUnits(TypedDict):
    action: float
    charge: float
    mass: float
    permittivity: float

hartree_atomic_units = AtomicUnits( action=1, charge =1, mass = 1, permittivity = 1)

"""Class representing a wavefunction in quantum mechanics.

Attributes:
    grid (np.array): The spatial grid.
    value (np.array): The wavefunction values on the grid.
"""
class Wavefunction(TypedDict):
    grid: np.array
    value: np.array

"""Class representing a particle in quantum mechanics.

Attributes:
    charge (float): The charge of the particle.
    mass (float): The mass of the particle.
    wavenumber (float): The wavenumber of the particle.
    state (Wavefunction): The wavefunction representing the state of the particle.
"""
class Particle(TypedDict):
    charge: float
    mass: float
    angular_frequency: float
    wavenumber: float
    state: Wavefunction

def mask_wavefunction(wavefunction: Wavefunction, mask: np.ndarray) -> Wavefunction:
    """
    Apply a mask to the wavefunction to store only certain points and their corresponding values.

    Parameters:
    wavefunction (Wavefunction): The original wavefunction.
    mask (np.ndarray): The mask that determines which points to keep.

    Returns:
    Wavefunction: The masked wavefunction.
    """
    masked_grid = wavefunction['grid'][mask]
    masked_value = wavefunction['value'][mask]
    return Wavefunction(grid=masked_grid, value=masked_value)

def probability_density(state: Wavefunction):
    return (np.conj(state["value"])*state["value"]).real

# def probability_current(state,dx):  
#     return np.imag(np.conj(state) * np.gradient(state,dx,axis=0))

def probability_current(state: Wavefunction):
    grid=state["grid"]
    dx=grid[1]-grid[0]
    return np.imag(np.conj(state["value"]) * np.gradient(state["value"],dx,axis=0))

# class ConstT:

#     def __init__(self,dx: float,\
#                   hbar: float = hartree_atomic_units["action"],\
#                   m: float = hartree_atomic_units["mass"]):
#         self.__t=(hbar**2)/(2*m*(dx**2))

#     @property
#     def __call__(self) -> float:
#         return self.__t

def const_t(dx: float,hbar: float=hartree_atomic_units["action"],m: float=hartree_atomic_units["mass"]) -> float:
    return (hbar**2)/(2*m*(dx**2))

# def const_t(hb: float,p: Particle):
#     dx=p.state.grid[1]-p.state.grid[0]
#     hbar=hb
#     mass=p.mass
#     return const_t(dx,hbar,mass)

# def const_t(au: AtomicUnits,p: Particle):
#     hbar=au.action
#     return const_t(hbar,p)

"""Class for calculating dispersion relations.

Attributes:
    unified_step_size (float): The step size for the calculations.
    action (float): The action unit.
    mass (float): The mass unit.
"""
class Dispersion:

    def __init__(self,unified_step_size: float,\
                  action: float = hartree_atomic_units["action"],\
                  mass: float = hartree_atomic_units["mass"]):

        self.__hbar=action
        self.__step_size=unified_step_size
        self.__t=const_t(unified_step_size,action,mass)

    """Convert energy to wave number (k).

    Args:
        energy (float): The energy value.

    Returns:
        float: The corresponding wave number (k).
    """
    def energy2k(self,energy: float) -> float:
        k=(1/self.__step_size)*np.arccos(1-(0.5*energy/self.__t))
        return k
    
    """Convert wave number (k) to energy.

    Args:
        k (float): The wave number.

    Returns:
        float: The corresponding energy.
    """
    def k2energy(self,k: float) -> float:
        E=2*self.__t*(1-np.cos(k*self.__step_size))
        return E
    
    """Convert wave number (k) to angular frequency (omega).

    Args:
        k (float): The wave number.

    Returns:
        float: The corresponding angular frequency.
    """
    def k2omega(self,k: float) -> float:
        return self.k2energy(k)/self.__hbar


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
        self.__ham_diags[1] += self.__charge*self.__spot(time)

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
                mass=hartree_atomic_units["mass"],\
                hbar=hartree_atomic_units["action"],\
                param_x0 = 0, param_lambda0 = 0.05,\
                param_theta0 = 0.4 ):

    dx = uxgrid[1]-uxgrid[0]
    xnum = len(uxgrid)
    lamb = param_lambda0

    if param_x0 == 0: 
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
def stationary_time_evolution(omega: float, times: np.array, eigenstate: Wavefunction, mask: Optional[np.array] = None) -> Wavefunction:
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
    mask : Optional[np.ndarray], optional
        A boolean mask array to apply to the grid and value of the eigenstate. If None, no mask is applied.

    Returns:
    --------
    Wavefunction
        A Wavefunction object containing the grid and the time-evolved values at each time point.
    """

    if mask is None:
        nx=len(eigenstate["grid"])
        nt=len(times)
        psi = np.zeros((nx,nt),dtype=np.complex128)

        eigenstate_value = eigenstate["value"]
        psi[:,0] = eigenstate_value
        for i, t in enumerate(times[1:], start=1):
            psi[:,i] = np.exp(-1j*omega*t)*eigenstate_value

        return Wavefunction(grid=eigenstate["grid"], value=psi)

    else:
        masked_grid = eigenstate['grid'][mask]
        masked_value = eigenstate['value'][mask]
        masked_nx=len(masked_grid)
        nt=len(times)
        masked_psi = np.zeros((masked_nx,nt),dtype=np.complex128)

        masked_psi[:,0] = masked_value
        for i, t in enumerate(times[1:], start=1):
            masked_psi[:,i] = np.exp(-1j*omega*t)*masked_value

        return Wavefunction(grid=masked_grid, value=masked_psi)



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

    Returns:
        np.array: The updated state wavefunction.
    """
    def step_n(self,n=2):
        for i in range(n):
            self.step_one()
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
        self.__psi0_initial=psi0["value"]
        self.__omega=omega

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

"""Class for calculating the time evolution of a wavefunction using split operator methods.

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
"""
class SplitTimeEvolutionCalculator:
    def __init__(self,psi0_omega: float, psi0_initial: Wavefunction,\
                 scalarpot=pots.ZeroPotential,vectorpot=pots.ZeroPotential,\
                 me=hartree_atomic_units["mass"],\
                 hbar=hartree_atomic_units["action"],\
                 charge=hartree_atomic_units["charge"],\
                 dt=0.01, t_start=0.0, t_stop=1.0, save_interval=0.1,\
                 mask: Optional[np.ndarray] = None):
        
        self.__psi0_omega=psi0_omega
        self.__psi0_initial=psi0_initial
        self.__utgrid=np.arange(t_start,t_stop,dt)
        self.__uxgrid=scalarpot.grid #psi1_initial["grid"]
        zeropot=pots.ZeroPotential(self.__uxgrid)
        self.__save_interval = save_interval

        hham2h1 = HamiltonOperator(self.__uxgrid, scalarpot=scalarpot,\
                                    vectorpot=vectorpot,\
                                    me=me, hbar=hbar, charge=charge )
        h0ham2h1 = HamiltonOperator(self.__uxgrid, scalarpot=scalarpot,\
                                    vectorpot=zeropot,\
                                    me=me, hbar=hbar, charge=charge )
        hham = HamiltonOperator(self.__uxgrid, scalarpot=scalarpot,\
                                    vectorpot=vectorpot,\
                                    me=me, hbar=hbar, charge=charge )
        v_cap=vcap_generator(self.__uxgrid)
        hham.vcap(v_cap)

        self.__nx=len(self.__uxgrid)
        self.__nt=len(self.__utgrid)
        self.__psi1_time_evolution=np.zeros((self.__nx,self.__nt),\
                                            dtype=np.complex128)#np.zeros_like(ini_state)

        self.__solver=CntdSes(np.zeros(self.__nx,dtype=np.complex128), hham,dt=dt)
        sb=SplitBoundary( hham2h1, h0ham2h1, psi0_initial,psi0_omega)
        self.__solver.boundary = sb

    """Run the time evolution calculation."""
    def run(self):

        self.__psi0_time_evolution = eigenstate_time_evolution(self.__psi0_omega,\
                                        self.__utgrid, self.__psi0_initial)

        self.__psi1_time_evolution[:,0]=np.zeros(self.__nx,dtype=np.complex128)
        save_step = int(self.__save_interval / self.__solver._CntdSes__dt)
        for i in range(1,self.__nt):
            if i % save_step == 0:
                self.__psi1_time_evolution[:,i]=self.__solver.step_one()
            else:
                self.__solver.step_one()
        # for i in range(1,self.__nt):
        #     self.__psi1_time_evolution[:,i]=self.__solver.step_one()

    @property
    def psi0_time_evolution(self):
        return self.__psi0_time_evolution
    
    @property
    def psi1_time_evolution(self):
        return Wavefunction(grid=self.__uxgrid,value=self.__psi1_time_evolution)
    
    @property
    def psi_time_evolution(self):
        self.__psi = self.__psi1_time_evolution + self.__psi0_time_evolution["value"]
        return Wavefunction(grid=self.__uxgrid,value=self.__psi)
