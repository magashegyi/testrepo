import quantum_toolkit as qtk
import scipy.sparse as sparse
import numpy as np
import scipy
from typing import TypedDict

class QuasiParticle(qtk.Particle,TypedDict):
    amplitudes: tuple
    transmission: float
    reflection: float

def retarded_green_function(ein: float,ham: qtk.HamiltonOperator):
    
    disp=qtk.Dispersion(ham.stepsize,ham.hbar,ham.mass)
    kin=disp.energy2k(ein)

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

# def amplitudes(qp: QuasiParticle):

#     return amplitudes_from_wavefunction(qp["wavenumber"],qp["state"])

def transmission(k: float,wf: qtk.Wavefunction):
    Ain,At = input_amplitude(k,wf),transmission_amplitude(k,wf)#amplitudes(qp)
    return np.abs(At*np.conj(At))/np.abs(Ain*np.conj(Ain))

def reflection(k: float,wf: qtk.Wavefunction):
    Ain,Ar = input_amplitude(k,wf),reflection_amplitude(k,wf)
    return np.abs(Ar*np.conj(Ar))/np.abs(Ain*np.conj(Ain))

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
    ain_pk=input_amplitude(kin,statepk)
    ain_mk=input_amplitude(-kin,statemk)

    nstatepk_value, nstatemk_value = statepk["value"]/np.abs(ain_pk), statemk["value"]/np.abs(ain_mk)
    nstatepk=qtk.Wavefunction(grid=ham.grid,value=nstatepk_value)
    nstatemk=qtk.Wavefunction(grid=ham.grid,value=nstatemk_value)

    if return_green:
        return nstatepk, nstatemk, green
    else:
        return nstatepk, nstatemk

def quasiparticles_at_given_energy(ein: float,kin: float,ham: qtk.HamiltonOperator, return_green=False):

    nstatepk, nstatemk, green = normed_states_at_given_energy(ein,kin,ham,return_green=True)

    qppk= QuasiParticle( charge=ham.charge, mass=ham.mass,\
            state=nstatepk,energy=ein, wavenumber=kin,\
            amplitudes = amplitudes(kin,nstatepk),\
            transmission = transmission(kin,nstatepk),\
            reflection = reflection(kin,nstatepk))
    
    qpmk= QuasiParticle(charge=ham.charge, mass=ham.mass,\
            state=nstatemk,energy=ein, wavenumber=-kin,\
            amplitudes = amplitudes(-kin,nstatemk),\
            transmission = transmission(-kin,nstatemk),\
            reflection = reflection(-kin,nstatemk))

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