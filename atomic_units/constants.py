from .base_units import AtomicBaseUnits
from .derived_units import AtomicUnitSystem
hartree_atomic_base_units = AtomicBaseUnits()
hartree_atomic_unit_system = AtomicUnitSystem(base_units=hartree_atomic_base_units)