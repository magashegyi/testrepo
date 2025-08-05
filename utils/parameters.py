import numpy as np
import pint
import atomic_units

def laser_parameters_converter(
    spot_size: pint.Quantity, magnitude: pint.Quantity, wavelength: pint.Quantity, num_cycles: float, beta: float, unit_sys: atomic_units.AtomicUnitSystem
):
    """
    spot_size: pint.Quantity (pl. 400 * ureg.nm)
    magnitude: pint.Quantity (pl. 1.0 * ureg('GV/m'))
    wavelength: pint.Quantity (pl. 800 * ureg.nm)
    num_cycles: float
    beta: float
    unit_sys: atomic_units.AtomicUnitSystem példány
    """
    spot_size_au = spot_size.to("nm").magnitude * unit_sys.convert_length_from("nm")
    s_min = -0.5 * spot_size_au
    s_max = 0.5 * spot_size_au

    central_wavelength_au = wavelength.to("nm").magnitude * unit_sys.convert_length_from("nm")
    omega = 2 * np.pi * unit_sys.speed_of_light / wavelength.to("nm").magnitude

    t_min = 0
    t_max = num_cycles * 2 * np.pi / omega

    amplitude_au = magnitude.to("GV/m").magnitude * unit_sys.convert_electric_field_from("GV/m")

    return dict(
        spatial_min=s_min,
        spatial_max=s_max,
        temporal_min=t_min,
        temporal_max=t_max,
        noc=num_cycles,
        amplitude=amplitude_au,
        beta=beta,
        omega=omega,
        central_wavelength=central_wavelength_au
    )