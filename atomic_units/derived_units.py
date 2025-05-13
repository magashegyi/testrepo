"""
Derived units
Author: István Magashegyi
Date: 2025 03 21
Description: This module contains various functions and constants for atomic units.
"""

"""
Rescaling Atomic units

Notation: 
One can say m=5kg then {m} = 5 and [m] = kg. In other words, { } means value and [ ] means unit.

HAU = Hartree Atomic Units
RAU = Rescaled Atomic Units

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
from .base_units import AtomicBaseUnits
from typing import Union
import pint
import math

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
        :param xk: Value of modified permittivity kappa0 = 4*Pi*epsilon0 in this atomic units.
        :param base_units: An instance of AtomicBaseUnits.
        """
        
        if base_units:
            super().__init__(xh=base_units.hb, xe=base_units.e0, xm=base_units.me, xk=base_units.k0)
        else:
            if xh is None or xe is None or xm is None or xk is None:
                raise ValueError("All units must be provided if base_units is not used")
            super().__init__(xh=xh, xe=xe, xm=xm, xk=xk)

        self._calculate_length_unit()
        self._calculate_energy_unit()
        self._calculate_velocity_unit()
        self._calculate_time_unit()
        self._calculate_frequency_unit()
        self._calculate_wavenumber_unit()
        self._calculate_electric_field_unit()
        self._calculate_electric_potential_unit()
        self._calculate_speed_of_light()
        self._calculate_fine_structure_constant()

        # Check if the fine structure constant is correct
        if not math.isclose(self._ralpha, 1.0, rel_tol=1e-12):
            raise ValueError(f"Fine structure constant is not correct: {self._ralpha} != {1}")
        
    def _calculate_length_unit(self):
        #Bohr radius -> atomic length unit
        a0 = self.ureg("bohr").to_base_units()
        self._ra = self._rk * self._rh**2 / (self._rm * self._re**2)
        self._length_unit = self._ra*a0

    def _calculate_energy_unit(self):
        # Hartree energy -> atomic energy unit
        eh = self.ureg("hartree").to("J")
        #self._ren = self._rm*self._re**4/((self._rk*self._rh)**2)
        self._ren = self._rh**2 / (self._rm * self._ra**2)
        self._energy_unit = self._ren*eh
 
    def _calculate_velocity_unit(self):
        # Velocity -> atomic velocity unit
        # a0 = (κ_0 * ħ^2) / (m_e * e_0^2)
        a0 = self.ureg("bohr").to_base_units()
        eh = self.ureg("hartree").to("J")
        avu = (a0*eh/self.ureg("hbar")).to("m/s") #atomic velovity unit
        self._rv = self._ra * self._ren / self._rh
        self._velocity_unit = self._rv*avu

    def _calculate_time_unit(self):
        # Time
        eh = self.ureg("hartree").to("J")
        ta=(self.ureg("hbar")/eh).to("s")
        self._rt = self._rh / self._ren
        self._time_unit = self._rt*ta

    def _calculate_frequency_unit(self):
        # Frequency
        self._frequency_unit = (1.0 / self._time_unit).to(self.ureg.hertz)

    def _calculate_wavenumber_unit(self):
        # Wavenumber
        #self._wavenumber_unit = (2.0*math.pi / self._length_unit).to(self.ureg("1/m"))
        self._wavenumber_unit = (1.0 / self._length_unit).to(self.ureg("1/m"))

    def _calculate_electric_field_unit(self):
        # Electric field unit
        self._electric_field_unit = (self._energy_unit / (self.charge_unit * self.length_unit)).to(self.ureg.volt / self.ureg.meter)
    
    def _calculate_electric_potential_unit(self):
        # Electric potential unit
        self._electric_potential_unit = (self._energy_unit / self.charge_unit).to(self.ureg.volt)

    def _calculate_speed_of_light(self):
        # Speed of light
        self._speed_of_light = (self.ureg("speed_of_light").to("m/s") / self._velocity_unit.to("m/s")).magnitude
        #print(f"Speed of light = {self._speed_of_light}")

    def _calculate_fine_structure_constant(self):
        # Fine structure constant
        self._ralpha = self._re**2 / (self._rk * self._rh * self._rv)
        #print(f"ralpha = {self._ralpha}")
        

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
    def wavenumber_unit(self) -> pint.Quantity:
        """Get the wavenumber unit."""
        return self._wavenumber_unit
    
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
    def fine_structure_constant(self) -> pint.Quantity:
        """Get the fine structure constant."""
        return self._ralpha*self.ureg("fine_structure_constant").to_base_units()
    
    def convert_length_from(self, unit: str) -> float:
        return (self.ureg(unit)/self.length_unit.to(unit)).magnitude
    
    def convert_energy_from(self, unit: str) -> float:
        return (self.ureg(unit)/self.energy_unit.to(unit)).magnitude

    def convert_time_from(self, unit: str) -> float:
        return (self.ureg(unit)/self.time_unit.to(unit)).magnitude
    
    def convert_electric_field_from(self, unit: str) -> float:
        return (self.ureg(unit)/self.electric_field_unit.to(unit)).magnitude

    def convert_wavenumber_from(self, unit: str) -> float:
        return (self.ureg(unit)/self.wavenumber_unit.to(unit)).magnitude

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