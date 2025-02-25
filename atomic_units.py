"""
Atomic units module
Author: István Magashegyi
Date: 2025 01 25
Description: This module contains various functions and constants for atomic units.
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
        hbv (float): value of modified planck constant (hbar) in this atomic units.
        e0v (float): value of elementary charge (e0) in this atomic units.
        mev (float): value of electron mass (me) in this atomic units.
        k0v (float): value of modified permittivity kappa0 = 4*Pi*epsilon0 in this atomic units.
    """
    def __init__(self, hbv: float = 1.0, e0v: float = 1.0, mev: float = 1.0, k0v: float = 1.0):
        """
        Initialize the AtomicBaseUnits class with the given parameters.

        :param hbv: Value of modified planck constant (hbar) in this atomic units.
        :param e0v: Value of elementary charge (e0) in this atomic units.
        :param mev: Value of electron mass (me) in this atomic units.
        :param k0v: Value of modified permittivity kappa0 = 4*Pi*epsilon0 in this atomic units.
        """

        self._hb = self._validate_positive(hbv, "value of modified planck constant (hbv)")
        self._e0 = self._validate_positive(e0v, "value of elementary charge (e0v)")
        self._me = self._validate_positive(mev, "value of electron mass (mev)")
        self._k0 = self._validate_positive(k0v, "value of modified permittivity (k0v)")

        self._rfa = 1.0/(self._hb) # rescale factor for action
        self._rfe = 1.0/(self._e0) # rescale factor for charge
        self._rfm = 1.0/(self._me) # rescale factor for mass
        self._rfk = 1.0/(self._k0) # rescale factor for permittivity

        self._ureg = pint.UnitRegistry()

        self._set_units()  

        self._check_fine_structure_constant()  

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
    
    def _set_units(self):
        """
        Set the units for the atomic base units.
        """
        self._action_unit = self._rfa * self._ureg("hbar").to_base_units()
        self._charge_unit = self._rfe * self._ureg("elementary_charge").to("C")
        self._mass_unit = self._rfm * self._ureg("electron_mass").to("kg")
        self._permittivity_unit = self._rfk * 4 * math.pi * self._ureg("vacuum_permittivity").to("F/m")

    def _check_fine_structure_constant(self):
        """
        Check if the fine structure constant is correct in the rescaled unit system.

        :raises ValueError: If the fine structure constant is not correct.
        """
        # fine structure constant (alpha) times speed of light (c) must be 1 in atomic units
        alpha_times_c = self._e0 ** 2 / (self._k0 * self._hb)
        if not math.isclose(alpha_times_c, 1.0, rel_tol=1e-12):
            raise ValueError(f"Fine structure constant is not correct because alpha times c: {alpha_times_c} != {1}")

    def __str__(self) -> str:
        """String representation of the atomic base units."""
        return (f"Unit and value of atomic base units in this rescaled atomic units:\n"
                f"- action unit := \t{self.action_unit}\n"
                f"- modified planck constant hbar := \t{self.hbar}\n"
                f"- charge unit := \t{self.charge_unit}\n"
                f"- elementary charge e0:= \t{self.e0}\n"
                f"- mass unit := \t{self.mass_unit}\n"
                f"- electron mass me := \t{self.me}\n"
                f"- modified permittivity unit := \t{self.permittivity_unit}\n"
                f"- modified permittivity kappa0 := \t{self.kappa0}\n")

    @property
    def hbar(self) -> float:
        """Get the magnitede of the modified planck constant in this rescaled atomic units."""
        return self._hb
    
    @property
    def e0(self) -> float:
        """Get the magnitude of the elementary charge in this rescaled atomic units."""
        return self._e0
    
    @property
    def me(self) -> float:
        """Get the magnitude of the electron mass in this rescaled atomic units."""
        return self._me
    
    @property
    def kappa0(self) -> float:
        """Get the magnitude of the modified permittivity in this rescaled atomic units."""
        return self._k0

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
                 hb: Union[float, None] = None, 
                 e0: Union[float, None] = None,
                 me: Union[float, None] = None, 
                 k0: Union[float, None] = None,
                 base_units: Union[AtomicBaseUnits, None] = None):
        """
        Initialize the AtomicUnitSystem class with rescale factors or base units.

        :param hb: Value of modified planck constant (hbar) in this atomic units.
        :param e0: Value of elementary charge (e0) in this atomic units.
        :param me: Value of electron mass (me) in this atomic units.
        :param k0: Value of modified permittivity kappa0 = 4*Pi*epsilon0 in this atomic units.
        :param base_units: An instance of AtomicBaseUnits.
        """
        
        if base_units:
            super().__init__(hbv=base_units.hbar, e0v=base_units.e0, mev=base_units.me, k0v=base_units.kappa0)
        else:
            if hb is None or e0 is None or me is None or k0 is None:
                raise ValueError("All units must be provided if base_units is not used")
            super().__init__(hbv=hb, e0v=e0, mev=me, k0v=k0)


        #self._cr = (self.er ** 2 / (self.hbr * ALPHA * self.kappar)).to_base_units()
        #self._length_unit = self.ureg.Quantity(self.permittivity_unit * self.action_unit ** 2 / (self.mass_unit * self.charge_unit ** 2)).to(self.ureg.meter)
        self._length_unit = (self.permittivity_unit * self.action_unit ** 2 / (self.mass_unit * self.charge_unit ** 2)).to(self.ureg.meter)
        self._energy_unit = (self.mass_unit*( self.charge_unit ** 2 / (self.permittivity_unit * self.action_unit)) ** 2).to(self.ureg.joule)
        self._time_unit = (self.action_unit / self.energy_unit).to(self.ureg.second)
        self._frequency_unit = (1.0 / self._time_unit).to(self.ureg.hertz)
        self._velocity_unit = (self._length_unit * self._energy_unit / self.action_unit).to(self.ureg.meter / self.ureg.second)
        self._speed_of_light = (self.ureg("speed_of_light").to("m/s") / self._velocity_unit).magnitude
        self._electric_field_unit = (self._energy_unit / (self.charge_unit * self._length_unit)).to(self.ureg.volt / self.ureg.meter)
        self._electric_potential_unit = (self._energy_unit / self.charge_unit).to(self.ureg.volt)

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

    def __str__(self) -> str:
        """String representation of the atomic units."""
        return (f"{super().__str__()}\n"
                f"Unit and value of atomic units in this rescaled atomic units:\n"
                f"- length unit := \t{self.length_unit}\n"
                f"- energy unit := \t{self.energy_unit}\n"
                f"- time unit := \t{self.time_unit}\n"
                f"- frequency unit := \t{self.frequency_unit}\n"
                f"- velocity unit := \t{self.velocity_unit}\n"
                f"- speed of light := \t{self.speed_of_light}\n"
                f"- electric field unit := \t{self.electric_field_unit}\n"
                f"- electric potential unit := \t{self.electric_potential_unit}\n")