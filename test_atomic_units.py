import unittest
from atomic_units import AtomicUnitSystem, AtomicBaseUnits

class TestAtomicUnitSystem(unittest.TestCase):

    def test_default_initialization(self):
        au = AtomicUnitSystem(action=1.0, charge=1.0, mass=1.0, permittivity=1.0)
        self.assertAlmostEqual(au.get_rescaled_m0_in_si().magnitude, 1.0 * au.ureg.kilogram.magnitude)
        self.assertAlmostEqual(au.get_rescaled_e0_in_si().magnitude, 1.0 * au.ureg.coulomb.magnitude)
        self.assertAlmostEqual(au.get_rescaled_hb_in_si().magnitude, 1.0 * au.ureg.joule * au.ureg.second.magnitude)
        self.assertAlmostEqual(au.get_rescaled_epsilon0_in_si().magnitude, 1.0 * au.ureg.farad / au.ureg.meter.magnitude)

    def test_custom_initialization(self):
        au = AtomicUnitSystem(action=2.0, charge=3.0, mass=4.0, permittivity=5.0)
        self.assertAlmostEqual(au.get_rescaled_m0_in_si().magnitude, 4.0 * au.ureg.kilogram.magnitude)
        self.assertAlmostEqual(au.get_rescaled_e0_in_si().magnitude, 3.0 * au.ureg.coulomb.magnitude)
        self.assertAlmostEqual(au.get_rescaled_hb_in_si().magnitude, 2.0 * au.ureg.joule * au.ureg.second.magnitude)
        self.assertAlmostEqual(au.get_rescaled_epsilon0_in_si().magnitude, 5.0 * au.ureg.farad / au.ureg.meter.magnitude)

    def test_set_rescale_factors(self):
        au = AtomicUnitSystem(action=1.0, charge=1.0, mass=1.0, permittivity=1.0)
        au.set_rescale_factor_of_m0(2.0)
        self.assertAlmostEqual(au.get_rescaled_m0_in_si().magnitude, 2.0 * au.ureg.kilogram.magnitude)
        au.set_rescale_factor_of_e0(3.0)
        self.assertAlmostEqual(au.get_rescaled_e0_in_si().magnitude, 3.0 * au.ureg.coulomb.magnitude)
        au.set_rescale_factor_of_hb(4.0)
        self.assertAlmostEqual(au.get_rescaled_hb_in_si().magnitude, 4.0 * au.ureg.joule * au.ureg.second.magnitude)
        au.set_rescale_factor_of_epsilon0(5.0)
        self.assertAlmostEqual(au.get_rescaled_epsilon0_in_si().magnitude, 5.0 * au.ureg.farad / au.ureg.meter.magnitude)

    def test_string_conversion(self):
        au = AtomicUnitSystem(action=2.0, charge=3.0, mass=4.0, permittivity=5.0)
        au_str = str(au)
        expected_str = ("Mass unit m0 := \t4.0 a.u. = \t" + str(4.0 * au.ureg.kilogram) + "\n"
                        "Charge unit e0 := \t3.0 a.u. = \t" + str(3.0 * au.ureg.coulomb) + "\n"
                        "Action unit hb := \t2.0 a.u. = \t" + str(2.0 * au.ureg.joule * au.ureg.second) + "\n"
                        "Permittivity epsilon0 := \t5.0 a.u. = \t" + str(5.0 * au.ureg.farad / au.ureg.meter) + "\n")
        self.assertEqual(au_str, expected_str)

    def test_from_string(self):
        input_str = ("Mass unit m0 := \t4.0 a.u. = \t" + str(4.0 * AtomicUnitSystem().ureg.kilogram) + "\n"
                     "Charge unit e0 := \t3.0 a.u. = \t" + str(3.0 * AtomicUnitSystem().ureg.coulomb) + "\n"
                     "Action unit hb := \t2.0 a.u. = \t" + str(2.0 * AtomicUnitSystem().ureg.joule * AtomicUnitSystem().ureg.second) + "\n"
                     "Permittivity epsilon0 := \t5.0 a.u. = \t" + str(5.0 * AtomicUnitSystem().ureg.farad / AtomicUnitSystem().ureg.meter) + "\n")
        au = AtomicUnitSystem.from_string(input_str)
        self.assertAlmostEqual(au.get_rescale_factor_of_m0(), 4.0)
        self.assertAlmostEqual(au.get_rescale_factor_of_e0(), 3.0)
        self.assertAlmostEqual(au.get_rescale_factor_of_hb(), 2.0)
        self.assertAlmostEqual(au.get_rescale_factor_of_epsilon0(), 5.0)

if __name__ == '__main__':
    unittest.main()
