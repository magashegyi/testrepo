"""
base units
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
        self._set_rescaled_units()  # Set the units for the atomic base units.

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

    def __str__(self) -> str:
        """String representation of the atomic base units."""
        return (f"Value of atomic base units in this rescaled atomic units:\n"
                f"- modified planck constant hbar := {self.hb}\n"
                f"- elementary charge := {self.e0}\n"
                f"- electron mass := {self.me}\n"
                f"- modified permittivity := {self.k0}\n")
    
    @property
    def hb(self) -> float:
        """Get the magnitede of the modified planck constant in this rescaled atomic units."""
        return self._xh
    
    @property
    def e0(self) -> float:
        """Get the magnitude of the elementary charge in this rescaled atomic units."""
        return self._xe
    
    @property
    def me(self) -> float:
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