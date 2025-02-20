"""
Atomic units module
Author: István Magashegyi
Date: 2025 01 25
Description: This module contains various functions and constants for atomic units.
"""

import re
from typing import Union
from constants import ALPHA
import pint
import math  # Import the math module

class AtomicBaseUnits:
    """
    Class representing atomic base units used in quantum mechanics.

    Attributes:
        action (float): The action unit.
        charge (float): The charge unit.
        mass (float): The mass unit.
        kappa0 (float): The permittivity unit.
    """
    def __init__(self, action: float, charge: float, mass: float, permittivity: float):
        """
        Initialize the AtomicBaseUnits class with the given parameters.

        :param action: Action unit.
        :param charge: Charge unit.
        :param mass: Mass unit.
        :param permittivity: Permittivity unit.
        """
        self.action = self._validate_positive(action, "action")
        self.charge = self._validate_positive(charge, "charge")
        self.mass = self._validate_positive(mass, "mass")
        self.permittivity = self._validate_positive(permittivity, "permittivity")

        self.ureg = pint.UnitRegistry()

        self.hbr = self.action * self.ureg("hbar").to_base_units()
        self.er = self.charge * self.ureg("elementary_charge").to("C")
        self.mr = self.mass * self.ureg("electron_mass").to("kg")
        # Permittivity -> kappa0 = 4*Pi*epsilon0
        self.kappar = self.permittivity * 4 * math.pi * self.ureg("vacuum_permittivity").to("F/m")

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

    @staticmethod
    def from_string(input_string: str) -> 'AtomicBaseUnits':
        """
        Create an AtomicBaseUnits instance from a string.

        :param input_string: The input string containing the rescale factors.
        :return: An instance of AtomicBaseUnits.
        """
        pattern = re.compile(r"^(.*):=\s*([0-9\.e\+-]+)\s+a\.u\.")
        check_strings = ["Mass unit m0 ", "Charge unit e0 ", "Action unit hb ", "Permittivity unit kappa0 "]

        values = {}
        lines = input_string.split('\n')
        for line in lines:
            match = pattern.match(line)
            if match:
                key = match.group(1).strip()
                value = float(match.group(2))
                if key == check_strings[0].strip():
                    values['mass'] = value
                elif key == check_strings[1].strip():
                    values['charge'] = value
                elif key == check_strings[2].strip():
                    values['action'] = value
                elif key == check_strings[3].strip():
                    values['permittivity'] = value

        required_keys = ['action', 'charge', 'mass', 'permittivity']
        for key in required_keys:
            if key not in values:
                raise ValueError(f"Missing required unit: {key}")

        return AtomicBaseUnits(action=values['action'], charge=values['charge'], mass=values['mass'], permittivity=values['permittivity'])

    @property
    def action_unit(self) -> pint.Quantity:
        """Get the action unit."""
        return self.hbr

    @property
    def charge_unit(self) -> pint.Quantity:
        """Get the charge unit."""
        return self.er
    
    @property
    def mass_unit(self) -> pint.Quantity:
        """Get the mass unit."""
        return self.mr
    
    @property
    def kappa0_unit(self) -> pint.Quantity:
        """Get the permittivity unit."""
        return self.kappar  
    
hartree_atomic_base_units = AtomicBaseUnits(action=1, charge=1, mass=1, permittivity=1)

class AtomicUnitSystem(AtomicBaseUnits):
    """
    Class to handle atomic units and their conversions.
    """
    def __init__(self, 
                 action: Union[float, None] = None, 
                 charge: Union[float, None] = None,
                 mass: Union[float, None] = None, 
                 permittivity: Union[float, None] = None,
                 base_units: Union[AtomicBaseUnits, None] = None):
        """
        Initialize the AtomicUnitSystem class with rescale factors or base units.

        :param action: Action unit in atomic units.
        :param charge: Charge unit in atomic units.
        :param mass: Mass unit in atomic units.
        :param permittivity: Permittivity unit in atomic units.
        :param base_units: An instance of AtomicBaseUnits.
        """
        
        if base_units:
            super().__init__(action=base_units.action, charge=base_units.charge, mass=base_units.mass, permittivity=base_units.permittivity)
        else:
            if action is None or charge is None or mass is None or permittivity is None:
                raise ValueError("All units must be provided if base_units is not used")
            super().__init__(action=action, charge=charge, mass=mass, permittivity=permittivity)

        self._cr = (self.er ** 2 / (self.hbr * ALPHA * self.kappar)).to_base_units()
        self._length_unit = (self.kappar * self.hbr ** 2 / (self.mr * self.er ** 2)).to(self.ureg.meter)
        self._energy_unit = (self.mr*( self.er ** 2 / (self.kappar * self.hbr)) ** 2).to(self.ureg.joule)
        self._time_unit = (self.hbr / self._energy_unit).to(self.ureg.second)
        self._frequency_unit = (1.0 / self._time_unit).to(self.ureg.hertz)
        self._velocity_unit = (self._length_unit * self._energy_unit / self.hbr).to(self.ureg.meter / self.ureg.second)
        self._speed_of_light = (self.ureg("speed_of_light").to("m/s") * self.ureg.meter / self.ureg.second / self._velocity_unit).to(self.ureg.meter / self.ureg.second)
        self._electric_field_unit = (self._energy_unit / (self.er * self._length_unit)).to(self.ureg.volt / self.ureg.meter)
        self._electric_potential_unit = (self._energy_unit / self.er).to(self.ureg.volt)

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
    def electric_field_unit(self) -> pint.Quantity:
        """Get the electric field unit."""
        return self._electric_field_unit

    @property
    def electric_potential_unit(self) -> pint.Quantity:
        """Get the electric potential unit."""
        return self._electric_potential_unit

    def __str__(self) -> str:
        """String representation of the atomic units."""
        return (f"Mass unit m0 := \t{self.mass} a.u. = \t{self.mr}\n"
                f"Charge unit e0 := \t{self.charge} a.u. = \t{self.er}\n"
                f"Action unit hb := \t{self.action} a.u. = \t{self.hbr}\n"
                f"Kappa0 unit := \t{self.permittivity} a.u. = \t{self.kappar}\n")

    @staticmethod
    def from_string(input_string: str) -> 'AtomicUnitSystem':
        """
        Create an AtomicUnits instance from a string.

        :param input_string: The input string containing the rescale factors.
        :return: An instance of AtomicUnits.
        """
        base_units = AtomicBaseUnits.from_string(input_string)
        return AtomicUnitSystem(base_units=base_units)