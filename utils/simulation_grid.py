import numpy as np

class Grid:
    """
    Represents a 1D grid with associated metadata.
    """
    def __init__(self, points=None, step=None, width=None, start=None, stop=None, center=True):
        """
        Initialize the grid using flexible parameter combinations.

        Parameters:
            points (int, optional): Number of grid points.
            step (float, optional): Step size.
            width (float, optional): Total width of the grid.
            start (float, optional): Start value of the grid.
            stop (float, optional): Stop value of the grid.
            center (bool, optional): Whether to center the grid around zero.
        """
        # Initialize grid parameters
        if points is not None and step is not None:
            num = int(points)
            dx = float(step)
            grid_width = num * dx
            grid_start = -grid_width / 2 if center else 0 if start is None else start
            grid_stop = grid_start + grid_width if stop is None else stop
        elif width is not None and step is not None:
            grid_width = float(width)
            dx = float(step)
            num = int(np.round(grid_width / dx))
            grid_start = -grid_width / 2 if center else 0 if start is None else start
            grid_stop = grid_start + grid_width if stop is None else stop
        elif width is not None and points is not None:
            grid_width = float(width)
            num = int(points)
            dx = grid_width / num
            grid_start = -grid_width / 2 if center else 0 if start is None else start
            grid_stop = grid_start + grid_width if stop is None else stop
        elif start is not None and stop is not None and points is not None:
            num = int(points)
            dx = (stop - start) / num
            grid_width = stop - start
            grid_start = start
            grid_stop = stop
        else:
            raise ValueError("Invalid grid parameters!")

        # Generate grid
        if center:
            grid = np.linspace(-grid_width / 2, grid_width / 2, num, endpoint=False)
        else:
            grid = np.linspace(grid_start, grid_stop, num, endpoint=False)

        # Store grid metadata
        self.grid = grid
        self.dx = dx
        self.num = num
        self.width = grid_width
        self.start = grid_start
        self.stop = grid_stop

class SimulationGrid:
    """
    Utility class for creating and managing spatial and temporal grids for simulations.
    Supports flexible initialization using various combinations of grid parameters.
    Also provides mask arrays for subsampling the spatial and temporal grids.
    """
    def __init__(
        self,
        x_points=None, x_step=None, x_width=None, x_start=None, x_stop=None,
        t_points=None, t_step=None, t_width=None, t_start=None, t_stop=None,
        spatial_mask_points=None,
        temporal_mask_points=None
    ):
        """
        Initialize the SimulationGrid.

        Parameters:
            x_points (int, optional): Number of spatial grid points.
            x_step (float, optional): Step size for the spatial grid.
            x_width (float, optional): Total width of the spatial grid.
            x_start (float, optional): Start value for the spatial grid.
            x_stop (float, optional): Stop value for the spatial grid.
            t_points (int, optional): Number of temporal grid points.
            t_step (float, optional): Step size for the temporal grid.
            t_width (float, optional): Total width of the temporal grid.
            t_start (float, optional): Start value for the temporal grid.
            t_stop (float, optional): Stop value for the temporal grid.
            spatial_mask_points (int, optional): Desired number of points in the spatial mask array.
            temporal_mask_points (int, optional): Desired number of points in the temporal mask array.
        """
        # Spatial grid
        self.spatial_grid = Grid(
            points=x_points, step=x_step, width=x_width, start=x_start, stop=x_stop, center=True
        )
        # Temporal grid
        self.temporal_grid = Grid(
            points=t_points, step=t_step, width=t_width, start=t_start, stop=t_stop, center=False
        )
        if temporal_mask_points is not None:
            self.spatial_mask_arr = self._create_mask(self.spatial_grid.num, desired_points=spatial_mask_points)
        else:
            self.spatial_mask_arr = None
        if temporal_mask_points is not None:
            self.temporal_mask_arr = self._create_mask(self.temporal_grid.num, desired_points=temporal_mask_points)
        else:
            self.temporal_mask_arr = None

    def _create_mask(self, grid_num, desired_points=2048):
        """
        Create a boolean mask array for subsampling a grid.

        Parameters:
            grid_num (int): Number of points in the grid.
            desired_points (int): Desired number of True values in the mask.

        Returns:
            mask_arr (np.ndarray): Boolean array with True at selected indices.
        """
        step_size = max(1, grid_num // max(1, desired_points))
        mask_arr = np.zeros(grid_num, dtype=bool)
        mask_arr[::step_size] = True
        return mask_arr

    @property
    def x(self):
        """Térbeli grid (numpy array)."""
        return self.spatial_grid.grid

    @property
    def t(self):
        """Időbeli grid (numpy array)."""
        return self.temporal_grid.grid

    @property
    def x_masked(self):
        """Maszkolt térbeli grid (numpy array vagy None)."""
        if self.spatial_mask_arr is not None:
            return self.spatial_grid.grid[self.spatial_mask_arr]
        return None

    @property
    def t_masked(self):
        """Maszkolt időbeli grid (numpy array vagy None)."""
        if self.temporal_mask_arr is not None:
            return self.temporal_grid.grid[self.temporal_mask_arr]
        return None

    @property
    def dx(self):
        """Térbeli lépésköz."""
        return self.spatial_grid.dx

    @property
    def dt(self):
        """Időbeli lépésköz."""
        return self.temporal_grid.dx

    @property
    def x_min(self):
        """Térbeli grid minimum értéke."""
        return self.spatial_grid.grid.min()

    @property
    def x_max(self):
        """Térbeli grid maximum értéke."""
        return self.spatial_grid.grid.max()

    @property
    def t_min(self):
        """Időbeli grid minimum értéke."""
        return self.temporal_grid.grid.min()

    @property
    def t_max(self):
        """Időbeli grid maximum értéke."""
        return self.temporal_grid.grid.max()

    @property
    def x_width(self):
        """Térbeli grid szélessége."""
        return self.spatial_grid.width

    @property
    def t_width(self):
        """Időbeli grid szélessége."""
        return self.temporal_grid.width

    @property
    def x_points(self):
        """Térbeli pontok száma."""
        return self.spatial_grid.num

    @property
    def t_points(self):
        """Időbeli pontok száma."""
        return self.temporal_grid.num

    @property
    def x_masked_points(self):
        """Maszkolt térbeli pontok száma (vagy None)."""
        if self.spatial_mask_arr is not None:
            return np.count_nonzero(self.spatial_mask_arr)
        return None

    @property
    def t_masked_points(self):
        """Maszkolt időbeli pontok száma (vagy None)."""
        if self.temporal_mask_arr is not None:
            return np.count_nonzero(self.temporal_mask_arr)
        return None

    # Additional utility functions for grid metadata, etc., can be added here.
