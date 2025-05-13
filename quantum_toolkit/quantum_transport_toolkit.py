import quantum_toolkit as qtk
import scipy.sparse as sparse
from numpy import linalg
import numpy as np
import scipy
from typing import TypedDict
from utils.smooth_utils import smoothing1D

class QuasiParticle(qtk.Particle,TypedDict):
    amplitudes: tuple
    transmission: float
    reflection: float

def retarded_green_function(ein: float,ham: qtk.HamiltonOperator):
    
    disp=qtk.CosineDispersion(ham.stepsize,ham.hbar,ham.mass)
    kin=disp.wavenumber(ein)

    ham0 = ham(0).toarray()

    Sigmaleft = np.zeros_like(ham0)
    Sigmaright = np.zeros_like(ham0)

    t=qtk.const_t(ham.stepsize,ham.hbar,ham.mass)

    Sigmaleft[0][0] = -t*(np.exp(1j*kin*ham.stepsize))
    Sigmaright[-1][-1] = -t*(np.exp(1j*kin*ham.stepsize))

    en=ein*sparse.identity(ham.grid.size).toarray()
    Gtoinv=en - ham0 - Sigmaleft - Sigmaright

    #retarded green function
    Gret=scipy.linalg.inv(Gtoinv)

    return Gret

# def input_amplitude(qp: QuasiParticle):

#     x1_index=2
#     x2_index=12

#     x1=qp.state.grid[x1_index]
#     x2=qp.state.grid[x2_index]

#     psi1=qp.state[x1_index]
#     psi2=qp.state[x2_index]

#     k=qp.wavenumber

#     Ain=(psi1*np.exp(1j*k*x1)-psi2*np.exp(1j*k*x2))/(np.exp(1j*k*2*x1)-np.exp(1j*k*2*x2))

#     return Ain

def input_amplitude(k: float,wf: qtk.Wavefunction):

    if k>0:
        x1_index=2
        x2_index=12

        x1=wf["grid"][x1_index]
        x2=wf["grid"][x2_index]

        psi1=wf["value"][x1_index]
        psi2=wf["value"][x2_index]

        Ain=(psi1*np.exp(1j*k*x1)-psi2*np.exp(1j*k*x2))/(np.exp(1j*k*2*x1)-np.exp(1j*k*2*x2))

    else:
        x1_index=-12
        x2_index=-2

        x1=wf["grid"][x1_index]
        x2=wf["grid"][x2_index]

        psi1=wf["value"][x1_index]
        psi2=wf["value"][x2_index]

        Ain=(psi1*np.exp(1j*k*x1)-psi2*np.exp(1j*k*x2))/(np.exp(1j*k*2*x1)-np.exp(1j*k*2*x2))


    return Ain

def transmission_amplitude(k: float,wf: qtk.Wavefunction):

    if k>0:

        At=wf["value"][-1]/np.exp(1j*k*wf["grid"][-1])
    else:

        At=wf["value"][0]/np.exp(1j*k*wf["grid"][0])

    return At

def reflection_amplitude(k: float,wf: qtk.Wavefunction):

    if k>0:
        x1_index=2
        x2_index=12

        x1=wf["grid"][x1_index]
        x2=wf["grid"][x2_index]

        psi1=wf["value"][x1_index]
        psi2=wf["value"][x2_index]

        Ar=(psi1*np.exp(-1j*k*x1)-psi2*np.exp(-1j*k*x2))/(np.exp(-1j*k*2*x1)-np.exp(-1j*k*2*x2))

    else:
        x1_index=-12
        x2_index=-2

        x1=wf["grid"][x1_index]
        x2=wf["grid"][x2_index]

        psi1=wf["value"][x1_index]
        psi2=wf["value"][x2_index]

        Ar=(psi1*np.exp(-1j*k*x1)-psi2*np.exp(-1j*k*x2))/(np.exp(-1j*k*2*x1)-np.exp(-1j*k*2*x2))

    return Ar

def amplitudes(k: float,wf: qtk.Wavefunction):

    if k>0:
        x1_index=2
        x2_index=12

        x1=wf["grid"][x1_index]
        x2=wf["grid"][x2_index]

        psi1=wf["value"][x1_index]
        psi2=wf["value"][x2_index]

        Ar=(psi1*np.exp(-1j*k*x1)-psi2*np.exp(-1j*k*x2))/(np.exp(-1j*k*2*x1)-np.exp(-1j*k*2*x2))

        #print(np.abs(Ar*np.conj(Ar)))

        Ain=(psi1*np.exp(1j*k*x1)-psi2*np.exp(1j*k*x2))/(np.exp(1j*k*2*x1)-np.exp(1j*k*2*x2))

        At=wf["value"][-1]/np.exp(1j*k*wf["grid"][-1])
    else:
        x1_index=-12
        x2_index=-2

        x1=wf["grid"][x1_index]
        x2=wf["grid"][x2_index]

        psi1=wf["value"][x1_index]
        psi2=wf["value"][x2_index]

        Ar=(psi1*np.exp(-1j*k*x1)-psi2*np.exp(-1j*k*x2))/(np.exp(-1j*k*2*x1)-np.exp(-1j*k*2*x2))

        #print(np.abs(Ar*np.conj(Ar)))

        Ain=(psi1*np.exp(1j*k*x1)-psi2*np.exp(1j*k*x2))/(np.exp(1j*k*2*x1)-np.exp(1j*k*2*x2))

        At=wf["value"][0]/np.exp(1j*k*wf["grid"][0])

    return (Ain,At,Ar)

def amplitudes2(k: float,wf: qtk.Wavefunction,fit_num: int = 100):
    """
    Compute the amplitudes of the incoming and reflected waves
    for a given wavefunction at a given energy. The amplitudes
    are computed by fitting the wavefunction to the sum of
    incoming and reflected waves.        
    """
    # Split the wavefunction into left and right parts

    x_left = wf["grid"][:fit_num]
    x_right = wf["grid"][-fit_num:]

    # Solve with least-squares solution
    A = np.zeros([fit_num,2],dtype=np.complex128)
    A[:,0]=np.exp(1j*k*x_left) # incoming wave on the left
    A[:,1]=np.exp(-1j*k*x_left) # reflected wave on the left
    b=wf["value"][:fit_num].T

    x_lstsq = linalg.lstsq(A,b,rcond=None)[0] # computing the numpy solution
    a_left=x_lstsq[0] # complex amplitude of positive propagating wave on the left
    b_left=x_lstsq[1] # complex amplitude of negative propagating wave on the left
    
    A = np.zeros([fit_num,2],dtype=np.complex128)
    A[:,0]=np.exp(1j*k*x_right) # incoming wave on the right
    A[:,1]=np.exp(-1j*k*x_right) # reflected wave on the right
    b=wf["value"][-fit_num:].T

    x_lstsq = linalg.lstsq(A,b,rcond=None)[0] # computing the numpy solution
    a_right=x_lstsq[0] # complex amplitude of positive propagating wave on the right
    b_right=x_lstsq[1] # complex amplitude of negative propagating wave on the right

    return (a_left,b_left,a_right,b_right)

def input_amplitude2(k: float,wf: qtk.Wavefunction, fit_num: int = 100):
    """
    Compute the amplitude of the incoming wave for a given wavefunction
    """

    a_left,b_left,a_right,b_right = amplitudes2(k,wf,fit_num)
    if k>0:
        return a_left
    else:
        return b_right
    
def transmission_amplitude2(k: float,wf: qtk.Wavefunction, fit_num: int = 100):
    """
    Compute the amplitude of the transmitted wave for a given wavefunction
    """

    a_left,b_left,a_right,b_right = amplitudes2(k,wf,fit_num)
    if k>0:
        return a_right
    else:
        return b_left


def reflection_amplitude2(k: float,wf: qtk.Wavefunction, fit_num: int = 100):
    """
    Compute the amplitude of the reflected wave for a given wavefunction
    """

    a_left,b_left,a_right,b_right = amplitudes2(k,wf,fit_num)
    if k>0:
        return b_left
    else:
        return a_right


# def amplitudes(qp: QuasiParticle):

#     return amplitudes_from_wavefunction(qp["wavenumber"],qp["state"])

def transmission(k: float,wf: qtk.Wavefunction):
    Ain,At = input_amplitude(k,wf),transmission_amplitude(k,wf)#amplitudes(qp)
    return np.abs(At*np.conj(At))/np.abs(Ain*np.conj(Ain))

def transmission2(k: float,wf: qtk.Wavefunction,fit_num: int = 100):
    """
    Compute the transmission coefficient for a given wavefunction at a given energy
    """
    a_left,b_left,a_right,b_right = amplitudes2(k,wf,fit_num)
    if k>0:
        return np.abs(a_right*np.conj(a_right))/np.abs(a_left*np.conj(a_left))
    else:
        return np.abs(b_left*np.conj(b_left))/np.abs(b_right*np.conj(b_right))

def reflection(k: float,wf: qtk.Wavefunction):
    """
    Compute the reflection coefficient for a given wavefunction at a given energy
    """
    Ain,Ar = input_amplitude(k,wf),reflection_amplitude(k,wf)
    return np.abs(Ar*np.conj(Ar))/np.abs(Ain*np.conj(Ain))

def reflection2(k: float,wf: qtk.Wavefunction,fit_num: int = 100):
    """
    Compute the reflection coefficient for a given wavefunction at a given energy
    """
    a_left,b_left,a_right,b_right = amplitudes2(k,wf,fit_num)
    if k>0:
        return np.abs(b_left*np.conj(b_left))/np.abs(a_left*np.conj(a_left))
    else:
        return np.abs(a_right*np.conj(a_right))/np.abs(b_right*np.conj(b_right))

def density_of_states_from_green_function(Gret):
    return (-1/np.pi)*np.imag(np.trace(Gret))


def density_of_states_at_given_energy(ein: float,ham: qtk.HamiltonOperator):

    #Generate retarded green function
    Gret=retarded_green_function(ein,ham)

    return density_of_states_from_green_function(Gret)

def spectral_function_from_green_function(Gret):

    return (-1)*np.imag(Gret).diagonal()

def spectral_function_at_given_energy(ein: float,ham: qtk.HamiltonOperator):

    #Generate retarded green function
    Gret=retarded_green_function(ein,ham)

    return spectral_function_from_green_function(Gret)

def states_from_green_function(Gret):

    statepk=Gret[:,0]
    statemk=Gret[:,-1]

    return statepk, statemk

def states_at_given_energy(ein: float,ham: qtk.HamiltonOperator, return_green=False):

    #Generate retarded green function
    green=retarded_green_function(ein,ham)

    statepk,statemk=states_from_green_function(green)

    if return_green:
        return qtk.Wavefunction(grid=ham.grid,value=statepk), qtk.Wavefunction(grid=ham.grid,value=statemk), green
    else:
        return qtk.Wavefunction(grid=ham.grid,value=statepk), qtk.Wavefunction(grid=ham.grid,value=statemk)

def normed_states_at_given_energy(ein: float,kin: float,ham: qtk.HamiltonOperator, return_green=False):
    statepk, statemk, green = states_at_given_energy(ein,ham,return_green=True)
    ain_pk=input_amplitude2(kin,statepk)
    ain_mk=input_amplitude2(-kin,statemk)

    nstatepk_value, nstatemk_value = statepk["value"]/np.abs(ain_pk), statemk["value"]/np.abs(ain_mk)
    nstatepk=qtk.Wavefunction(grid=ham.grid,value=nstatepk_value)
    nstatemk=qtk.Wavefunction(grid=ham.grid,value=nstatemk_value)

    if return_green:
        return nstatepk, nstatemk, green
    else:
        return nstatepk, nstatemk
    
def normed_states_at_given_energy2(ein: float,kin: float,ham: qtk.HamiltonOperator, return_green=False):
    statepk, statemk, green = states_at_given_energy(ein,ham,return_green=True)
    ain_pk=input_amplitude2(kin,statepk)
    ain_mk=input_amplitude2(-kin,statemk)

    #nstatepk_value, nstatemk_value = statepk["value"]/np.abs(ain_pk), statemk["value"]/np.abs(ain_mk)
    nstatepk=qtk.Wavefunction(grid=statepk["grid"],value=statepk["value"]/np.abs(ain_pk))
    nstatemk=qtk.Wavefunction(grid=statemk["grid"],value=statemk["value"]/np.abs(ain_mk))
    print(np.abs(ain_pk),np.abs(ain_mk))

    if return_green:
        return nstatepk, nstatemk, green
    else:
        return nstatepk, nstatemk

def quasiparticles_at_given_energy(ein: float, kin: float, ham: qtk.HamiltonOperator, return_green=False, normed=True):

    if normed:
        statepk, statemk, green = normed_states_at_given_energy2(ein,kin,ham,return_green=True)
    else:
        statepk, statemk, green = states_at_given_energy(ein,ham,return_green=True)

    qppk= QuasiParticle( charge=ham.charge, mass=ham.mass,\
            state=statepk,energy=ein, wavenumber=kin,\
            amplitudes = amplitudes(kin,statepk),\
            transmission = transmission(kin,statepk),\
            reflection = reflection(kin,statepk))
    
    qpmk= QuasiParticle(charge=ham.charge, mass=ham.mass,\
            state=statemk,energy=ein, wavenumber=-kin,\
            amplitudes = amplitudes(-kin,statemk),\
            transmission = transmission(-kin,statemk),\
            reflection = reflection(-kin,statemk))

    if return_green:
        return qppk, qpmk, green
    else:
        return qppk, qpmk

# def quasiparticles_at_given_energy(ein: float, ham: qtk.HamiltonOperator, charge: float=1,\
#                                     mass: float=1, action: float = 1):
    
#     statepk, statemk = states_at_given_energy(ein,ham)

#     dx=ham.grid[1]-ham.grid[0]
#     disp=qtk.Dispersion(dx,action,mass)
#     kin=disp.energy2k(ein)

#     qp_pk = QuasiParticle( charge=1, mass=1, state=statepk, \
#                             energy=ein, wavenumber=kin )
#     qp_mk = QuasiParticle( charge=1, mass=1,state=statemk, \
#                                 energy=ein, wavenumber=kin )

#     Ain_pk,At_pk,Ar_pk = trat.amplitudes(qp_pk)
#     Ain_mk,At_mk,Ar_mk = trat.amplitudes(qp_mk)


#     print(np.abs(Ain_pk),np.abs(At_pk),np.abs(Ar_pk))
#     print(np.abs(Ain_mk),np.abs(At_mk),np.abs(Ar_mk))

#     return statepk, statemk