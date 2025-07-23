import numpy as np
from typing import Optional, Union, Tuple
from enum import Enum
import h5py


class GridInitMode(Enum):
    """Enumeration for different grid initialization modes."""
    POINTS_STEP = "points_step"
    WIDTH_STEP = "width_step"
    WIDTH_POINTS = "width_points"
    START_STOP_POINTS = "start_stop_points"


class Grid:
    """
    Represents a 1D grid with associated metadata.
    
    This class provides a flexible way to create 1D grids using various parameter combinations.
    It supports centered and non-centered grids, and automatically calculates derived properties.
    """
    
    def __init__(
        self, 
        points: Optional[int] = None, 
        step: Optional[float] = None, 
        width: Optional[float] = None, 
        start: Optional[float] = None, 
        stop: Optional[float] = None, 
        center: bool = True
    ):
        """
        Initialize the grid using flexible parameter combinations.

        Parameters:
            points: Number of grid points.
            step: Step size between grid points.
            width: Total width of the grid.
            start: Start value of the grid.
            stop: Stop value of the grid.
            center: Whether to center the grid around zero.
            
        Raises:
            ValueError: If parameter combination is invalid or insufficient.
        """
        self._validate_parameters(points, step, width, start, stop)
        
        # Determine initialization mode and calculate grid parameters
        mode = self._determine_init_mode(points, step, width, start, stop)
        grid_params = self._calculate_grid_parameters(mode, points, step, width, start, stop, center)
        
        # Generate the grid array
        self.grid = self._generate_grid_array(grid_params, center)
        
        # Store metadata
        self.dx = grid_params['dx']
        self._num = grid_params['num']
        self._width = grid_params['width']
        self.start = grid_params['start']
        self.stop = grid_params['stop']
    
    def _validate_parameters(
        self, 
        points: Optional[int], 
        step: Optional[float], 
        width: Optional[float], 
        start: Optional[float], 
        stop: Optional[float]
    ) -> None:
        """Validate input parameters."""
        if points is not None and points <= 0:
            raise ValueError("Number of points must be positive.")
        if step is not None and step <= 0:
            raise ValueError("Step size must be positive.")
        if width is not None and width <= 0:
            raise ValueError("Width must be positive.")
        if start is not None and stop is not None and start >= stop:
            raise ValueError("Start value must be less than stop value.")
    
    def _determine_init_mode(
        self, 
        points: Optional[int], 
        step: Optional[float], 
        width: Optional[float], 
        start: Optional[float], 
        stop: Optional[float]
    ) -> GridInitMode:
        """Determine which initialization mode to use based on provided parameters."""
        if points is not None and step is not None:
            return GridInitMode.POINTS_STEP
        elif width is not None and step is not None:
            return GridInitMode.WIDTH_STEP
        elif width is not None and points is not None:
            return GridInitMode.WIDTH_POINTS
        elif start is not None and stop is not None and points is not None:
            return GridInitMode.START_STOP_POINTS
        else:
            raise ValueError(
                "Invalid parameter combination. Please provide one of: "
                "(points, step), (width, step), (width, points), or (start, stop, points)."
            )
    
    def _calculate_grid_parameters(
        self, 
        mode: GridInitMode, 
        points: Optional[int], 
        step: Optional[float], 
        width: Optional[float], 
        start: Optional[float], 
        stop: Optional[float], 
        center: bool
    ) -> dict:
        """Calculate grid parameters based on initialization mode."""
        if mode == GridInitMode.POINTS_STEP:
            return self._calc_points_step(points, step, start, stop, center)
        elif mode == GridInitMode.WIDTH_STEP:
            return self._calc_width_step(width, step, start, stop, center)
        elif mode == GridInitMode.WIDTH_POINTS:
            return self._calc_width_points(width, points, start, stop, center)
        elif mode == GridInitMode.START_STOP_POINTS:
            return self._calc_start_stop_points(start, stop, points)
        else:
            raise ValueError(f"Unknown initialization mode: {mode}")
    
    def _calc_points_step(
        self, 
        points: int, 
        step: float, 
        start: Optional[float], 
        stop: Optional[float], 
        center: bool
    ) -> dict:
        """Calculate parameters for points+step initialization."""
        num = int(points)
        dx = float(step)
        grid_width = num * dx
        grid_start = -grid_width / 2 if center else (0 if start is None else start)
        grid_stop = grid_start + grid_width if stop is None else stop
        
        return {
            'num': num,
            'dx': dx,
            'width': grid_width,
            'start': grid_start,
            'stop': grid_stop
        }
    
    def _calc_width_step(
        self, 
        width: float, 
        step: float, 
        start: Optional[float], 
        stop: Optional[float], 
        center: bool
    ) -> dict:
        """Calculate parameters for width+step initialization."""
        grid_width = float(width)
        dx = float(step)
        num = int(np.round(grid_width / dx))
        grid_start = -grid_width / 2 if center else (0 if start is None else start)
        grid_stop = grid_start + grid_width if stop is None else stop
        
        return {
            'num': num,
            'dx': dx,
            'width': grid_width,
            'start': grid_start,
            'stop': grid_stop
        }
    
    def _calc_width_points(
        self, 
        width: float, 
        points: int, 
        start: Optional[float], 
        stop: Optional[float], 
        center: bool
    ) -> dict:
        """Calculate parameters for width+points initialization."""
        grid_width = float(width)
        num = int(points)
        dx = grid_width / num
        grid_start = -grid_width / 2 if center else (0 if start is None else start)
        grid_stop = grid_start + grid_width if stop is None else stop
        
        return {
            'num': num,
            'dx': dx,
            'width': grid_width,
            'start': grid_start,
            'stop': grid_stop
        }
    
    def _calc_start_stop_points(
        self, 
        start: float, 
        stop: float, 
        points: int
    ) -> dict:
        """Calculate parameters for start+stop+points initialization."""
        num = int(points)
        dx = (stop - start) / num
        grid_width = stop - start
        
        return {
            'num': num,
            'dx': dx,
            'width': grid_width,
            'start': start,
            'stop': stop
        }
    
    def _generate_grid_array(self, params: dict, center: bool) -> np.ndarray:
        """Generate the actual grid array."""
        if center:
            return np.linspace(-params['width'] / 2, params['width'] / 2, params['num'], endpoint=False)
        else:
            return np.linspace(params['start'], params['stop'], params['num'], endpoint=False)

    @property
    def delta(self) -> float:
        """Step size of the grid."""
        return self.dx

    @property
    def min(self) -> float:
        """Minimum value of the grid."""
        return self.start

    @property
    def max(self) -> float:
        """Maximum value of the grid."""
        return self.stop

    @property
    def width(self) -> float:
        """Width of the grid."""
        return self._width

    @property
    def num(self) -> int:
        """Number of grid points."""
        return self._num
    
    @property
    def values(self) -> np.ndarray:
        """Get the grid values as a numpy array."""
        return self.grid
    
    def __len__(self) -> int:
        """Return the number of grid points."""
        return self._num
    
    def __getitem__(self, key: Union[int, slice]) -> Union[float, np.ndarray]:
        """Enable indexing and slicing of the grid."""
        return self.grid[key]
    
    def __repr__(self) -> str:
        """Return string representation of the grid."""
        return (f"Grid(num={self.num}, dx={self.dx:.4f}, "
                f"width={self.width:.4f}, start={self.start:.4f}, stop={self.stop:.4f})")

    def to_hdf5_group(self, group):
        """
        Save the Grid object to an HDF5 group.
        
        Parameters:
            group: HDF5 group to save the grid data to.
        """
        # Save basic grid properties as attributes
        group.attrs["num"] = self._num
        group.attrs["dx"] = self.dx
        group.attrs["width"] = self._width
        group.attrs["start"] = self.start
        group.attrs["stop"] = self.stop
        
        # Save the actual grid array as a dataset
        group.create_dataset("grid", data=self.grid)
    
    @classmethod
    def from_hdf5_group(cls, group):
        """
        Load a Grid object from an HDF5 group.
        
        Parameters:
            group: HDF5 group containing the grid data.
            
        Returns:
            Grid: A new Grid object with the loaded data.
        """
        # Load the grid array
        grid_data = group["grid"][()]
        
        # Load metadata
        num = group.attrs["num"]
        dx = group.attrs["dx"]
        width = group.attrs["width"]
        start = group.attrs["start"]
        stop = group.attrs["stop"]
        
        # Create a new Grid object and manually set its properties
        # We use a dummy initialization and then override the values
        grid_obj = cls.__new__(cls)
        grid_obj.grid = grid_data
        grid_obj.dx = dx
        grid_obj._num = num
        grid_obj._width = width
        grid_obj.start = start
        grid_obj.stop = stop
        
        return grid_obj
        

class MaskGenerator:
    """Utility class for generating various types of masks for grid subsampling."""
    
    @staticmethod
    def uniform_mask(grid_num: int, desired_points: int = 2048) -> np.ndarray:
        """
        Create a uniform boolean mask array for subsampling a grid.

        Parameters:
            grid_num: Number of points in the grid.
            desired_points: Desired number of True values in the mask.

        Returns:
            Boolean array with True at uniformly spaced indices.
        """
        if desired_points >= grid_num:
            return np.ones(grid_num, dtype=bool)
        
        step_size = max(1, grid_num // desired_points)
        mask_arr = np.zeros(grid_num, dtype=bool)
        mask_arr[::step_size] = True
        return mask_arr
    
    @staticmethod
    def random_mask(grid_num: int, desired_points: int = 2048, seed: Optional[int] = None) -> np.ndarray:
        """
        Create a random boolean mask array for subsampling a grid.

        Parameters:
            grid_num: Number of points in the grid.
            desired_points: Desired number of True values in the mask.
            seed: Random seed for reproducibility.

        Returns:
            Boolean array with True at randomly selected indices.
        """
        if seed is not None:
            np.random.seed(seed)
        
        if desired_points >= grid_num:
            return np.ones(grid_num, dtype=bool)
        
        indices = np.random.choice(grid_num, size=desired_points, replace=False)
        mask_arr = np.zeros(grid_num, dtype=bool)
        mask_arr[indices] = True
        return mask_arr
    
    @staticmethod
    def range_mask(grid_num: int, grid_array: np.ndarray, start: float, stop: float) -> np.ndarray:
        """
        Create a mask for a specific range of values in the grid.

        Parameters:
            grid_num: Number of points in the grid.
            grid_array: The grid array to mask.
            start: Start value of the range.
            stop: Stop value of the range.

        Returns:
            Boolean array with True for values in the specified range.
        """
        mask_arr = np.zeros(grid_num, dtype=bool)
        mask_arr[(grid_array >= start) & (grid_array <= stop)] = True
        return mask_arr
    
    @staticmethod
    def center_mask(grid_num: int, grid_array: np.ndarray, center: float, width: float) -> np.ndarray:
        """
        Create a mask centered around a specific value with given width.

        Parameters:
            grid_num: Number of points in the grid.
            grid_array: The grid array to mask.
            center: Center value of the mask.
            width: Width of the mask region.

        Returns:
            Boolean array with True for values within the centered region.
        """
        half_width = width / 2
        return MaskGenerator.range_mask(grid_num, grid_array, center - half_width, center + half_width)


class GridAccessor:
    """
    A flexible accessor for grid data with optional masking capabilities.
    
    This class provides a clean interface for accessing grid data with various
    masking options while maintaining backward compatibility.
    """
    
    def __init__(self, grid: Grid, masks: dict):
        """
        Initialize the GridAccessor.
        
        Parameters:
            grid: The underlying Grid object.
            masks: Dictionary of named masks.
        """
        self._grid = grid
        self._masks = masks
    
    def __call__(self, mask: Union[str, np.ndarray, None] = None) -> np.ndarray:
        """
        Get grid values with optional masking.
        
        Parameters:
            mask: Either a mask name (str), a boolean array, or None for full grid.
            
        Returns:
            Grid values, optionally masked.
        """
        if mask is None:
            return self._grid.grid
        elif isinstance(mask, str):
            if mask not in self._masks:
                raise KeyError(f"Mask '{mask}' not found")
            return self._grid.grid[self._masks[mask]]
        elif isinstance(mask, np.ndarray):
            return self._grid.grid[mask]
        else:
            raise ValueError("mask must be a string (mask name), numpy array, or None")
    
    @property
    def values(self) -> np.ndarray:
        """Get the full grid values."""
        return self._grid.grid
    
    @property
    def grid(self) -> Grid:
        """Get the underlying Grid object."""
        return self._grid
    
    @property 
    def num(self) -> int:
        """Number of grid points."""
        return self._grid.num
    
    @property
    def dx(self) -> float:
        """Grid step size."""
        return self._grid.dx
        
    @property
    def width(self) -> float:
        """Grid width."""
        return self._grid.width
    
    @property
    def start(self) -> float:
        """Grid start value."""
        return self._grid.start
    
    @property
    def stop(self) -> float:
        """Grid stop value."""
        return self._grid.stop
    
    def __len__(self) -> int:
        """Return the number of grid points."""
        return len(self._grid)
    
    def __getitem__(self, key: Union[int, slice]) -> Union[float, np.ndarray]:
        """Enable indexing and slicing of the grid."""
        return self._grid[key]
    
    def __repr__(self) -> str:
        """Return string representation."""
        available_masks = list(self._masks.keys())
        return f"GridAccessor({self._grid}, masks={available_masks})"


class SimulationGrid:
    """
    Simplified utility class for creating and managing spatial and temporal grids for simulations.
    
    This class provides a clean interface for creating simulation grids with flexible masking
    capabilities. Masks can be applied directly through the grid properties.
    """
    
    def __init__(
        self,
        # Spatial grid parameters
        x_points: Optional[int] = None, 
        x_step: Optional[float] = None, 
        x_width: Optional[float] = None, 
        x_start: Optional[float] = None, 
        x_stop: Optional[float] = None,
        # Temporal grid parameters
        t_points: Optional[int] = None, 
        t_step: Optional[float] = None, 
        t_width: Optional[float] = None, 
        t_start: Optional[float] = None, 
        t_stop: Optional[float] = None,
    ):
        """
        Initialize the SimulationGrid.

        Parameters:
            x_points: Number of spatial grid points.
            x_step: Step size for the spatial grid.
            x_width: Total width of the spatial grid.
            x_start: Start value for the spatial grid.
            x_stop: Stop value for the spatial grid.
            t_points: Number of temporal grid points.
            t_step: Step size for the temporal grid.
            t_width: Total width of the temporal grid.
            t_start: Start value for the temporal grid.
            t_stop: Stop value for the temporal grid.
        """
        # Create spatial grid (centered around zero)
        self._spatial_grid = Grid(
            points=x_points, step=x_step, width=x_width, 
            start=x_start, stop=x_stop, center=True
        )
        
        # Create temporal grid (not centered)
        self._temporal_grid = Grid(
            points=t_points, step=t_step, width=t_width, 
            start=t_start, stop=t_stop, center=False
        )
        
        # Initialize mask dictionaries for named masks
        self.masks = {
            'spatial': {},
            'temporal': {}
        }
    
    def add_mask(
        self, 
        grid_type: str,
        name: str, 
        mask_type: str = 'uniform', 
        desired_points: Optional[int] = None,
        start: Optional[float] = None,
        stop: Optional[float] = None,
        center: Optional[float] = None,
        width: Optional[float] = None,
        random_seed: Optional[int] = None
    ) -> None:
        """
        Add a named mask for either spatial or temporal grid.

        Parameters:
            grid_type: Either 'spatial' or 'temporal'.
            name: Name identifier for the mask.
            mask_type: Type of mask ('uniform', 'random', 'range', 'center').
            desired_points: Number of points for uniform/random masks.
            start: Start value for range mask.
            stop: Stop value for range mask.
            center: Center value for center mask.
            width: Width for center mask.
            random_seed: Random seed for random masks.
        """
        if grid_type not in ['spatial', 'temporal']:
            raise ValueError("grid_type must be 'spatial' or 'temporal'")
            
        grid = self._spatial_grid if grid_type == 'spatial' else self._temporal_grid
        mask = self._create_mask(
            grid.num, grid.grid, mask_type, desired_points, 
            start, stop, center, width, random_seed
        )
        self.masks[grid_type][name] = mask
    
    def _create_mask(
        self, 
        grid_num: int, 
        grid_array: np.ndarray,
        mask_type: str, 
        desired_points: Optional[int] = None,
        start: Optional[float] = None,
        stop: Optional[float] = None,
        center: Optional[float] = None,
        width: Optional[float] = None,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        """Create a mask array based on the specified type and parameters."""
        if mask_type == 'uniform':
            if desired_points is None:
                raise ValueError("desired_points is required for uniform mask")
            return MaskGenerator.uniform_mask(grid_num, desired_points)
        elif mask_type == 'random':
            if desired_points is None:
                raise ValueError("desired_points is required for random mask")
            return MaskGenerator.random_mask(grid_num, desired_points, random_seed)
        elif mask_type == 'range':
            if start is None or stop is None:
                raise ValueError("start and stop are required for range mask")
            return MaskGenerator.range_mask(grid_num, grid_array, start, stop)
        elif mask_type == 'center':
            if center is None or width is None:
                raise ValueError("center and width are required for center mask")
            return MaskGenerator.center_mask(grid_num, grid_array, center, width)
        else:
            raise ValueError(f"Unknown mask type: {mask_type}. Use 'uniform', 'random', 'range', or 'center'.")
    
    def remove_mask(self, grid_type: str, name: str) -> None:
        """Remove a named mask."""
        if grid_type not in ['spatial', 'temporal']:
            raise ValueError("grid_type must be 'spatial' or 'temporal'")
        if name in self.masks[grid_type]:
            del self.masks[grid_type][name]
        else:
            raise KeyError(f"Mask '{name}' not found in {grid_type} masks")
    
    def get_mask(self, grid_type: str, name: str) -> np.ndarray:
        """Get a named mask."""
        if grid_type not in ['spatial', 'temporal']:
            raise ValueError("grid_type must be 'spatial' or 'temporal'")
        if name in self.masks[grid_type]:
            return self.masks[grid_type][name]
        else:
            raise KeyError(f"Mask '{name}' not found in {grid_type} masks")
    
    def list_masks(self, grid_type: str) -> list:
        """List all mask names for a grid type."""
        if grid_type not in ['spatial', 'temporal']:
            raise ValueError("grid_type must be 'spatial' or 'temporal'")
        return list(self.masks[grid_type].keys())
    
    # Modern grid access properties with masking support
    @property 
    def spatial(self) -> 'GridAccessor':
        """Access spatial grid with optional masking."""
        return GridAccessor(self._spatial_grid, self.masks['spatial'])
    
    @property
    def temporal(self) -> 'GridAccessor':
        """Access temporal grid with optional masking."""
        return GridAccessor(self._temporal_grid, self.masks['temporal'])
    
    def get_grid_info(self) -> dict:
        """
        Get comprehensive information about both grids and all masks.
        
        Returns:
            Dictionary containing grid metadata and mask information.
        """
        spatial_mask_info = {}
        for name, mask in self.masks['spatial'].items():
            spatial_mask_info[name] = {
                'points': np.count_nonzero(mask),
                'total_points': len(mask),
                'coverage': np.count_nonzero(mask) / len(mask) * 100
            }
        
        temporal_mask_info = {}
        for name, mask in self.masks['temporal'].items():
            temporal_mask_info[name] = {
                'points': np.count_nonzero(mask),
                'total_points': len(mask),
                'coverage': np.count_nonzero(mask) / len(mask) * 100
            }
        
        return {
            'spatial': {
                'num': self._spatial_grid.num,
                'dx': self._spatial_grid.dx,
                'width': self._spatial_grid.width,
                'start': self._spatial_grid.start,
                'stop': self._spatial_grid.stop,
                'masks': spatial_mask_info
            },
            'temporal': {
                'num': self._temporal_grid.num,
                'dx': self._temporal_grid.dx,
                'width': self._temporal_grid.width,
                'start': self._temporal_grid.start,
                'stop': self._temporal_grid.stop,
                'masks': temporal_mask_info
            }
        }
    
    def __repr__(self) -> str:
        """Return string representation of the simulation grid."""
        spatial_mask_names = list(self.masks['spatial'].keys())
        temporal_mask_names = list(self.masks['temporal'].keys())
        
        return (f"SimulationGrid(\n"
                f"  spatial: {self._spatial_grid},\n"
                f"  temporal: {self._temporal_grid},\n"
                f"  spatial_masks: {spatial_mask_names},\n"
                f"  temporal_masks: {temporal_mask_names}\n"
                f")")

    def to_hdf5_group(self, group):
        """
        Save the SimulationGrid object to an HDF5 group.
        
        Parameters:
            group: HDF5 group to save the simulation grid data to.
        """
        # Save spatial grid
        spatial_grp = group.create_group("spatial_grid")
        self._spatial_grid.to_hdf5_group(spatial_grp)
        
        # Save temporal grid
        temporal_grp = group.create_group("temporal_grid")
        self._temporal_grid.to_hdf5_group(temporal_grp)
        
        # Save spatial masks
        if self.masks['spatial']:
            spatial_masks_grp = group.create_group("spatial_masks")
            for name, mask in self.masks['spatial'].items():
                spatial_masks_grp.create_dataset(name, data=mask)
        
        # Save temporal masks
        if self.masks['temporal']:
            temporal_masks_grp = group.create_group("temporal_masks")
            for name, mask in self.masks['temporal'].items():
                temporal_masks_grp.create_dataset(name, data=mask)
    
    @classmethod
    def from_hdf5_group(cls, group):
        """
        Load a SimulationGrid object from an HDF5 group.
        
        Parameters:
            group: HDF5 group containing the simulation grid data.
            
        Returns:
            SimulationGrid: A new SimulationGrid object with the loaded data.
        """
        # Load spatial and temporal grids
        spatial_grid = Grid.from_hdf5_group(group["spatial_grid"])
        temporal_grid = Grid.from_hdf5_group(group["temporal_grid"])
        
        # Create a new SimulationGrid object and manually set its properties
        sim_grid = cls.__new__(cls)
        sim_grid._spatial_grid = spatial_grid
        sim_grid._temporal_grid = temporal_grid
        sim_grid.masks = {'spatial': {}, 'temporal': {}}
        
        # Load spatial masks if they exist
        if "spatial_masks" in group:
            spatial_masks_grp = group["spatial_masks"]
            for name in spatial_masks_grp.keys():
                sim_grid.masks['spatial'][name] = spatial_masks_grp[name][()]
        
        # Load temporal masks if they exist
        if "temporal_masks" in group:
            temporal_masks_grp = group["temporal_masks"]
            for name in temporal_masks_grp.keys():
                sim_grid.masks['temporal'][name] = temporal_masks_grp[name][()]
        
        return sim_grid
