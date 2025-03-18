import numpy as np
from abc import ABC, abstractmethod # abstract base class

class Potential(ABC):
    """Abstract base class for potentials.

    Attributes:
        grid (np.array): The spatial grid on which the potential is defined.
    """

    def __init__(self, grid: np.array):
        self._grid = grid

    @property
    def grid(self) -> np.array:
        return self._grid

    @abstractmethod
    def __call__(self, time: float) -> np.array:
        pass

    @abstractmethod
    def value_at(self, space: float, time: float) -> float:
        pass

class ZeroPotential(Potential):
    """Class representing a zero potential."""

    def __call__(self, time: float) -> np.array:
        return np.zeros_like(self._grid)

    def value_at(self, space: float, time: float) -> float:
        return float(0.0)
    
class ConstantPotential(Potential):

    def __init__(self, grid: np.array, value: float):
        super().__init__(grid)
        self.__value = value

    """Class representing a zero potential."""

    def __call__(self, time: float) -> np.array:
        return self.__value*np.ones_like(self._grid)

    def value_at(self, space: float, time: float) -> float:
        return self.__value

class StackPotential(Potential):
    """Class representing a stack potential.

    Attributes:
        wallwidth (float): Width of the wall.
        wallheight (float): Height of the wall.
        wellwidth (float): Width of the well.
        welldepth (float): Depth of the well.
    """

    def __init__(self, grid: np.array, wallwidth=0.3, wallheight=0.2, wellwidth=12, welldepth=0.5):
        super(StackPotential, self).__init__(grid)
        self.__wallwidth = wallwidth
        self.__wallheight = wallheight
        self.__wellwidth = wellwidth
        self.__wellstartpos = -(wellwidth / 2) - wallwidth
        self.__wellstoppos = (wellwidth / 2) + wallwidth
        self.__welldepth = welldepth
    
    def __call__(self, time: float) -> np.array:
        leftshoulder = self.__wallheight * np.heaviside(self._grid - self.__wellstartpos * np.ones_like(self._grid), 1) - (self.__wallheight + self.__welldepth) * np.heaviside(self._grid - (self.__wellstartpos + self.__wallwidth) * np.ones_like(self._grid), 1)
        rightshoulder = (self.__wallheight + self.__welldepth) * np.heaviside(self._grid - (self.__wellstoppos - self.__wallwidth) * np.ones_like(self._grid), 0) - self.__wallheight * np.heaviside(self._grid - self.__wellstoppos * np.ones_like(self._grid), 1)

        return leftshoulder + rightshoulder

    def value_at(self, space: float, time: float) -> float:
        return float(0.0)
    
    @property
    def wallwidth(self) -> float:
        return self.__wallwidth

    @wallwidth.setter
    def wallwidth(self, value: float):
        self.__wallwidth = value
    
    @property
    def wellwidth(self) -> float:
        return self.__wellwidth
    
    @wellwidth.setter
    def wellwidth(self, value: float):
        self.__wellwidth = value
    
    @property
    def wallheight(self) -> float:
        return self.__wallheight
    
    @wallheight.setter
    def wallheight(self, value: float):
        self.__wallheight = value
    
    @property
    def welldepth(self) -> float:
        return self.__welldepth
    
    @welldepth.setter
    def welldepth(self, value: float):
        self.__welldepth = value

# def smoothing1D_psi(x):
#     x_half=np.where(x>0,x,1)
#     return np.where(x>0,np.exp(-1/x_half),0)


# def smoothing1D_phi(x,psi=None):
#     if psi is None:
#         psi=smoothing1D_psi

#     return np.where((x>0) & (x<1),psi(x)/(psi(x)+psi(1-x)),0)+np.where(x>=1,1,0)

# def smoothing1D(x,f,g,a=0,b=1,psi=None):
#     if b<a:
#         raise Exception("b has to be bigger than a") 

#     lx=(x-a)/(b-a)
#     return (1-smoothing1D_phi(lx,psi))*f(x)+smoothing1D_phi(lx,psi)*g(x)


from smooth_utils import smoothing1D

class SmoothStackPotential(Potential):
    """Generates a smooth stack potential profile with customizable well and wall properties.

    Attributes:
        wallwidth (float): Width of the wall.
        wallheight (float): Height of the wall.
        wellwidth (float): Width of the well.
        welldepth (float): Depth of the well.
        wallrise (float): Rise of the wall.
        wellfall (float): Fall of the well.
    """

    def __init__(self, grid: np.array, wallwidth=0.3, wallheight=0.2, wellwidth=12, welldepth=0.5, wallrise=0.2, wellfall=0.6):
        super().__init__(grid)
        self.__wallwidth = wallwidth
        self.__wallheight = wallheight
        self.__wallrise = wallrise
        self.__wellwidth = wellwidth
        self.__wellstartpos = -(wellwidth / 2)
        self.__wellstoppos = (wellwidth / 2)
        self.__welldepth = welldepth
        self.__wellfall = wellfall

    @staticmethod
    def leftwell(x: np.array, a: float, h: float, d: float, rf: float) -> np.array:
        f = lambda x: np.full_like(x, h)
        g = lambda x: np.full_like(x, -d)
        a_start = a - 0.5 * rf
        a_stop = a + 0.5 * rf
        return np.where(x < a_stop, smoothing1D(x, f, g, a_start, a_stop), -d)

    def _generate_walls(self) -> np.array:
        wall_start = -(self.__wellwidth / 2) - self.__wallwidth - 0.5 * self.__wallrise
        wall_stop = -(self.__wellwidth / 2) - self.__wallwidth + 0.5 * self.__wallrise
        g = lambda x: self.leftwell(x, self.__wellstartpos, self.__wallheight, self.__welldepth, self.__wellfall)
        return smoothing1D(self._grid, lambda x: np.zeros_like(x), g, a=wall_start, b=wall_stop)

    def __call__(self, time: float) -> np.array:
        wall_potential = self._generate_walls()
        return wall_potential + np.flip(wall_potential) + np.full_like(self.grid, self.welldepth)

    def value_at(self, space: float, time: float) -> float:
        return float(0.0)

    @property
    def wallwidth(self) -> float:
        return self.__wallwidth

    @wallwidth.setter
    def wallwidth(self, value: float):
        self.__wallwidth = value
    
    @property
    def wellwidth(self) -> float:
        return self.__wellwidth
    
    @wellwidth.setter
    def wellwidth(self, value: float):
        self.__wellwidth = value
    
    @property
    def wallheight(self) -> float:
        return self.__wallheight
    
    @wallheight.setter
    def wallheight(self, value: float):
        self.__wallheight = value
    
    @property
    def welldepth(self) -> float:
        return self.__welldepth
    
    @welldepth.setter
    def welldepth(self, value: float):
        self.__welldepth = value

class LaserPotential(Potential):
    """Class representing a laser potential.

    Attributes:
        spatial_min (float): Minimum spatial value.
        spatial_max (float): Maximum spatial value.
        amplitude (float): Amplitude of the laser potential.
        cep (float): Carrier envelope phase.
        noc (int): Number of cycles.
        temporal_min (float): Minimum temporal value.
        temporal_max (float): Maximum temporal value.
    """

    def __init__(self, spatial_grid: np.array, spatial_min=-5.0, spatial_max=5.0, amplitude=0.01, cep=0, noc=3, temporal_min=0.0, temporal_max=10.0):
        self.__spatial_min = spatial_min
        self.__spatial_max = spatial_max
        self.__spatial_width = spatial_max - spatial_min
        self.__spatial_grid = spatial_grid

        self.__temporal_min = temporal_min
        self.__temporal_max = temporal_max
        self.__temporal_width = temporal_max - temporal_min

        self.__amplitude = amplitude
        self.__noc = noc
        self.__cep = cep

        self.__omega0 = self.__noc * 2.0 * np.pi / self.__temporal_width
    
    def __call__(self, t: float) -> np.array:
        lp_x = self.__amplitude * (np.heaviside(self.__spatial_grid - self.__spatial_min * np.ones_like(self.__spatial_grid), 1) - np.heaviside(self.__spatial_grid - self.__spatial_max * np.ones_like(self.__spatial_grid), 1))

        lp_envelope = (np.heaviside(t, 1) - np.heaviside(t - self.__temporal_width, 1)) * pow(np.cos((np.pi / self.__temporal_width) * (t - 0.5 * self.__temporal_width)), 2)
        lp_t = lp_envelope * np.cos(self.__omega0 * (t - 0.5 * self.__temporal_width) + self.__cep)

        laserpot = lp_t * lp_x

        return laserpot

    def value_at(self, x: float, t: float) -> float:
        lp_x = self.__amplitude * (np.heaviside(x - self.__spatial_min * np.ones_like(x), 1) - np.heaviside(x - self.__spatial_max * np.ones_like(x), 1))

        lp_envelope = (np.heaviside(t, 1) - np.heaviside(t - self.__temporal_width, 1)) * pow(np.cos((np.pi / self.__temporal_width) * (t - 0.5 * self.__temporal_width)), 2)
        lp_t = lp_envelope * np.cos(self.__omega0 * (t - 0.5 * self.__temporal_width) + self.__cep)

        laserpot = lp_t * lp_x

        return laserpot

    @property
    def omega0(self) -> float:
        return self.__omega0
 
    @property
    def spatial_width(self) -> float:
        return self.__spatial_width
    
    @property
    def temporal_width(self) -> float:
        return self.__temporal_width
    
    @property
    def number_of_cycles(self) -> int:
        return self.__noc
    
    @property
    def carrier_envelope_phase(self) -> float:
        return self.__cep
    
    @property
    def temporal_min(self) -> float:
        return self.__temporal_min
    
    @property
    def temporal_max(self) -> float:
        return self.__temporal_max
    
    @property
    def spatial_min(self) -> float:
        return self.__spatial_min
    
    @property
    def spatial_max(self) -> float:
        return self.__spatial_max

from smooth_utils import smooth_jumper

class SmoothLaserPotential(Potential):
    """Class representing a smooth laser potential.

    Attributes:
        spatial_min (float): Minimum spatial value.
        spatial_max (float): Maximum spatial value.
        amplitude (float): Amplitude of the laser potential.
        cep (float): Carrier envelope phase.
        noc (int): Number of cycles.
        temporal_min (float): Minimum temporal value.
        temporal_max (float): Maximum temporal value.
        beta (float): Smoothing parameter.
    """

    def __init__(self, spatial_grid: np.array, spatial_min=-5.0, spatial_max=5.0, amplitude=0.01, cep=0, noc=3, temporal_min=0.0, temporal_max=10.0, beta=10):
        self.__spatial_min = spatial_min
        self.__spatial_max = spatial_max
        self.__spatial_width = spatial_max - spatial_min
        self.__spatial_grid = spatial_grid

        self.__temporal_min = temporal_min
        self.__temporal_max = temporal_max
        self.__temporal_width = temporal_max - temporal_min

        self.__amplitude = amplitude
        self.__noc = noc
        self.__cep = cep

        self.__beta = beta

        self.__omega0 = self.__noc * 2.0 * np.pi / self.__temporal_width
    
    def __call__(self, t: float) -> np.array:
        jumps = [
            (self.__spatial_min, 1, self.__beta),  # Upward jump at x=xmin, height=1, uses beta from variable
            (self.__spatial_max, -1, self.__beta) # downward jump at x=xmax, height=-1, uses beta from variable
        ]
        lp_x = self.__amplitude * smooth_jumper(self.__spatial_grid, jumps)

        lp_envelope = (np.heaviside(t, 1) - np.heaviside(t - self.__temporal_width, 1)) * pow(np.cos((np.pi / self.__temporal_width) * (t - 0.5 * self.__temporal_width)), 2)
        lp_t = lp_envelope * np.cos(self.__omega0 * (t - 0.5 * self.__temporal_width) + self.__cep)

        laserpot = lp_t * lp_x

        return laserpot

    def value_at(self, x: float, t: float) -> float:
        lp_x = self.__amplitude * (np.heaviside(x - self.__spatial_min * np.ones_like(x), 1) - np.heaviside(x - self.__spatial_max * np.ones_like(x), 1))

        lp_envelope = (np.heaviside(t, 1) - np.heaviside(t - self.__temporal_width, 1)) * pow(np.cos((np.pi / self.__temporal_width) * (t - 0.5 * self.__temporal_width)), 2)
        lp_t = lp_envelope * np.cos(self.__omega0 * (t - 0.5 * self.__temporal_width) + self.__cep)

        laserpot = lp_t * lp_x

        return laserpot

    @property
    def omega0(self) -> float:
        return self.__omega0
 
    @property
    def spatial_width(self) -> float:
        return self.__spatial_width
    
    @property
    def temporal_width(self) -> float:
        return self.__temporal_width
    
    @property
    def number_of_cycles(self) -> int:
        return self.__noc
    
    @property
    def carrier_envelope_phase(self) -> float:
        return self.__cep
    
    @property
    def temporal_min(self) -> float:
        return self.__temporal_min
    
    @property
    def temporal_max(self) -> float:
        return self.__temporal_max
    
    @property
    def spatial_min(self) -> float:
        return self.__spatial_min
    
    @property
    def spatial_max(self) -> float:
        return self.__spatial_max