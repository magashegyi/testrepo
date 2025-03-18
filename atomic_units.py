"""
Atomic units module
Author: István Magashegyi
Date: 2025 01 25
Description: This module contains various functions and constants for atomic units.
"""

"""
Rescaling Atomic units

Notation: 
One can say m=5kg then {m} = 5 and [m] = kg. In other words, { } means value and [ ] means unit.

HAU = Hartree Atomic Units
RAU = Rescaled Atomic Units

Atomic base units:

action: ħ = 1.054571817 * 10^-34 Js

{ħ}_{SI} = 1.054571817 * 10^-34; [ħ]_{SI} = Js
{ħ}_{HAU} = 1; [ħ]_{HAU} = 1.054571817 * 10^-34 Js
{ħ}_{RAU} = X_h; [ħ]_{RAU} = (1 / X_h) * 1.054571817 * 10^-34 Js

elementary charge: e_0 = 1.602176634 * 10^-19 C

{e_0}_{SI} = 1.602176634 * 10^-19; [e_0]_{SI} = C
{e_0}_{HAU} = 1; [e_0]_{HAU} = 1.602176634 * 10^-19 C
{e_0}_{RAU} = X_e; [e_0]_{RAU} = (1 / X_e) * 1.602176634 * 10^-19 C

electron rest mass: m_e = 9.1093837139(28) * 10^-31 kg

{m_e}_{SI} = 9.1093837139 * 10^-31; [m_e]_{SI} = kg
{m_e}_{HAU} = 1; [m_e]_{HAU} = 9.1093837139 * 10^-31 kg
{m_e}_{RAU} = X_m; [m_e]_{RAU} = (1 / X_m) * 9.1093837139 * 10^-31 kg

permittivity: κ_0 = 4π ε_0 = 1.11265005620 * 10^-10 F/m

{κ_0}_{SI} = 1.11265005620 * 10^-10; [κ_0]_{SI} = F/m
{κ_0}_{HAU} = 1; [κ_0]_{HAU} = 1.11265005620 * 10^-10 F/m
{κ_0}_{RAU} = X_k; [κ_0]_{RAU} = (1 / X_k) * 1.11265005620 * 10^-10 F/m

Derived units

1. example: length unit (Bohr radius)
a_0 = (κ_0 * ħ^2) / (m_e * e_0^2)
In SI units:
a_0 = (κ_0_SI * ħ_SI^2) / (m_e_SI * e_0_SI^2)
a_0 = (1.11265005620 * 10^-10 * (1.054571817 * 10^-34)^2) / (9.1093837139 * 10^-31 * (1.602176634 * 10^-19)^2)
a_0 = 5.2917720989499097 * 10^-11 meters
Therefore, in SI units:
{a_0}_{SI} = 5.2917720989499097 * 10^-11; [a_0]_{SI} = meters

In Hartree atomic units (HAU):
a_0 = (κ_0_HAU * ħ_HAU^2) / (m_e_HAU * e_0_HAU^2)
a_0 = (1 * 1^2) / (1 * 1^2)
a_0 = 1 a_0 = 1 l_HAU
Therefore, in HAU:
{a_0}_{HAU} = 1; [a_0]_{HAU} = 5.2917720989499097 * 10^-11 meters = a_0

In Rescaled Atomic Units (RAU):
a_0 = (κ_0_RAU * ħ_RAU^2) / (m_e_RAU * e_0_RAU^2)
a_0 = (X_k * X_h^2) / (X_m * X_e^2) * (r_e * 1.11265005620 * 10^-10 F/m * (r_h * 1.054571817 * 10^-34 Js)^2) / (r_m * 9.1093837139 * 10^-31 kg * (r_e * 1.602176634 * 10^-19 C)^2)
a_0 = (X_k * X_h^2) / (X_m * X_e^2) * (r_e * r_h^2) / (r_m * r_e^2) * a_0
a_0 = (X_k * X_h^2) / (X_m * X_e^2) l_RAU
Therefore, in RAU:
{a_0}_{RAU} = (X_k * X_h^2) / (X_m * X_e^2); [a_0]_{RAU} = (r_e * r_h^2) / (r_m * r_e^2) * a_0
"""

import re
from typing import Union
#from constants import ALPHA, CLIGHT
import pint
import math  # Import the math module

class AtomicBaseUnits:
    """
    Class representing atomic base units used in quantum mechanics.

    Attributes:
        xh (float): value of modified planck constant (hbar) in this atomic units.
        xe (float): value of elementary charge (xe) in this atomic units.
        xm (float): value of electron mass (xm) in this atomic units.
        xk (float): value of modified permittivity kappa0 = 4*Pi*epsilon0 in this atomic units.
    """
    def __init__(self, xh: float = 1.0, xe: float = 1.0, xm: float = 1.0, xk: float = 1.0):
        """
        Initialize the AtomicBaseUnits class with the given parameters.

        :param xh: Value of modified planck constant (hbar) in this atomic units.
        :param xe: Value of elementary charge (xe) in this atomic units.
        :param xm: Value of electron mass (xm) in this atomic units.
        :param xk: Value of modified permittivity (xm = 4*Pi*epsilon0) in this atomic units.
        """

        self._xh = self._validate_positive(xh, "value of modified planck constant (xh)")
        self._xe = self._validate_positive(xe, "value of elementary charge (xe)")
        self._xm = self._validate_positive(xm, "value of electron mass (xm)")
        self._xk = self._validate_positive(xk, "value of modified permittivity (k0)")

        self._rh = 1.0/(self._xh) # rescale factor for action
        self._re = 1.0/(self._xe) # rescale factor for charge
        self._rm = 1.0/(self._xm) # rescale factor for mass
        self._rk = 1.0/(self._xk) # rescale factor for permittivity

        self._ureg = pint.UnitRegistry()

        self._set_rescaled_units()  

        #self._check_fine_structure_constant()  

    @staticmethod
    def _validate_positive(value: float, name: str) -> float:
        """
        Validate that the given value is positive.

        :param value: The value to validate.
        :param name: The name of the value.
        :return: The validated value.
        """
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value
    
    def _set_rescaled_units(self):
        """
        Set the units for the atomic base units.
        """
        self._action_unit = self._rh * self._ureg("hbar").to_base_units()
        self._charge_unit = self._re * self._ureg("elementary_charge").to("C")
        self._mass_unit = self._rm * self._ureg("electron_mass").to("kg")
        self._permittivity_unit = self._rk * 4 * math.pi * self._ureg("vacuum_permittivity").to("F/m")

    def _check_fine_structure_constant(self):
        """
        Check if the fine structure constant is correct in the rescaled unit system.

        :raises ValueError: If the fine structure constant is not correct.
        """
        # fine structure constant (alpha) times speed of light (c) must be 1 in atomic units
        alpha_times_c = self._xe ** 2 / (self._xk * self._xh)
        if not math.isclose(alpha_times_c, 1.0, rel_tol=1e-12):
            raise ValueError(f"Fine structure constant is not correct because alpha times c: {alpha_times_c} != {1}")

    def __str__(self) -> str:
        """String representation of the atomic base units."""
        return (f"Value of atomic base units in this rescaled atomic units:\n"
                f"- modified planck constant hbar := {self.hbar}\n"
                f"- elementary charge := {self.e}\n"
                f"- electron mass := {self.m}\n"
                f"- modified permittivity := {self.k0}\n")
    
        #         """String representation of the atomic base units."""
        # return (f"Value of atomic base units in this rescaled atomic units:\n"
        #         f"- action unit := \t{self.action_unit}\n"
        #         f"- modified planck constant hbar := \t{self.hbar}\n"
        #         f"- charge unit := \t{self.charge_unit}\n"
        #         f"- elementary charge := \t{self.e}\n"
        #         f"- mass unit := \t{self.mass_unit}\n"
        #         f"- electron mass := \t{self.m}\n"
        #         f"- modified permittivity unit := \t{self.permittivity_unit}\n"
        #         f"- modified permittivity := \t{self.k0}\n")

    @property
    def hbar(self) -> float:
        """Get the magnitede of the modified planck constant in this rescaled atomic units."""
        return self._xh
    
    @property
    def e(self) -> float:
        """Get the magnitude of the elementary charge in this rescaled atomic units."""
        return self._xe
    
    @property
    def m(self) -> float:
        """Get the magnitude of the electron mass in this rescaled atomic units."""
        return self._xm
    
    @property
    def k0(self) -> float:
        """Get the magnitude of the modified permittivity in this rescaled atomic units."""
        return self._xk

    @property
    def ureg(self) -> pint.UnitRegistry:
        """Get the unit registry."""
        return self._ureg

    @property
    def action_unit(self) -> pint.Quantity:
        """Get the action unit."""
        return self._action_unit

    @property
    def charge_unit(self) -> pint.Quantity:
        """Get the charge unit."""
        return self._charge_unit
    
    @property
    def mass_unit(self) -> pint.Quantity:
        """Get the mass unit."""
        return self._mass_unit
    
    @property
    def permittivity_unit(self) -> pint.Quantity:
        """Get the permittivity unit."""
        return self._permittivity_unit 
    
hartree_atomic_base_units = AtomicBaseUnits()

class AtomicUnitSystem(AtomicBaseUnits):
    """
    Class to handle atomic units and their conversions.
    """
    def __init__(self, 
                 xh: Union[float, None] = None, 
                 xe: Union[float, None] = None,
                 xm: Union[float, None] = None, 
                 xk: Union[float, None] = None,
                 base_units: Union[AtomicBaseUnits, None] = None):
        """
        Initialize the AtomicUnitSystem class with rescale factors or base units.

        :param xh: Value of modified planck constant (hbar) in this atomic units.
        :param xe: Value of elementary charge (xe) in this atomic units.
        :param xm: Value of electron mass (xm) in this atomic units.
        :param k0: Value of modified permittivity kappa0 = 4*Pi*epsilon0 in this atomic units.
        :param base_units: An instance of AtomicBaseUnits.
        """
        
        if base_units:
            super().__init__(xh=base_units.hbar, xe=base_units.e, xm=base_units.m, xk=base_units.k0)
        else:
            if xh is None or xe is None or xm is None or xk is None:
                raise ValueError("All units must be provided if base_units is not used")
            super().__init__(xh=xh, xe=xe, xm=xm, xk=xk)

        #Bohr radius
        a0 = self.ureg("bohr").to_base_units()
        self._ra = self._rk * self._rh**2 / (self._rm * self._re**2)
        self._length_unit = self._ra*a0
        #print(f"ra = {self._ra}")

        # Hartree energy
        eh = self.ureg("hartree").to("J")
        self._ren = self._rm*self._re**4/((self._rk*self._rh)**2)
        self._energy_unit = self._ren*eh
        #print(f"ren = {self._ren}")

        # Velocity
        avu = (a0*eh/self.ureg("hbar")).to("m/s") #atomic velovity unit
        self._rv = self._ra * self._ren / self._rh
        self._velocity_unit = self._rv*avu
        #print(f"rv = {self._rv}")

        # Time
        ta=(self.ureg("hbar")/eh).to("s")
        self._rt = self._rh / self._ren
        self._time_unit = self._rt*ta
        #print(f"rt = {self._rt}")

        self._frequency_unit = (1.0 / self._time_unit).to(self.ureg.hertz)
        self._speed_of_light = (self.ureg("speed_of_light").to("m/s") / self._velocity_unit).magnitude
        self._electric_field_unit = (self._energy_unit / (self.charge_unit * self._length_unit)).to(self.ureg.volt / self.ureg.meter)
        self._electric_potential_unit = (self._energy_unit / self.charge_unit).to(self.ureg.volt)
        self._wavenumber = 2*math.pi/self._length_unit

        self._ralpha = self._re**2 / (self._rk * self._rh * self._rv)
        #print(f"ralpha = {self._ralpha}")
        if not math.isclose(self._ralpha, 1.0, rel_tol=1e-12):
            raise ValueError(f"Fine structure constant is not correct: {self._ralpha} != {1}")

    @property
    def length_unit(self) -> pint.Quantity:
        """Get the length unit."""
        return self._length_unit

    @property
    def energy_unit(self) -> pint.Quantity:
        """Get the energy unit."""
        return self._energy_unit
    
    @property
    def time_unit(self) -> pint.Quantity:
        """Get the time unit."""
        return self._time_unit

    @property
    def frequency_unit(self) -> pint.Quantity:
        """Get the frequency unit."""
        return self._frequency_unit

    @property
    def velocity_unit(self) -> pint.Quantity:
        """Get the velocity unit."""
        return self._velocity_unit
    
    @property
    def speed_of_light(self) -> float:
        """Get the speed of light."""
        return self._speed_of_light

    @property
    def electric_field_unit(self) -> pint.Quantity:
        """Get the electric field unit."""
        return self._electric_field_unit

    @property
    def electric_potential_unit(self) -> pint.Quantity:
        """Get the electric potential unit."""
        return self._electric_potential_unit
    
    @property
    def wavenumber_unit(self) -> pint.Quantity:
        """Get the wavenumber unit."""
        return self._wavenumber

    def __str__(self) -> str:
        """String representation of the atomic units."""
        return (f"{super().__str__()}\n"
                f"Derived units in this rescaled atomic unit system:\n"
                f"- length unit := \t{self.length_unit}\n"
                f"- energy unit := \t{self.energy_unit}\n"
                f"- time unit := \t{self.time_unit}\n"
                f"- frequency unit := \t{self.frequency_unit}\n"
                f"- velocity unit := \t{self.velocity_unit}\n"
                f"- speed of light := \t{self.speed_of_light}\n"
                f"- electric field unit := \t{self.electric_field_unit}\n"
                f"- electric potential unit := \t{self.electric_potential_unit}\n")