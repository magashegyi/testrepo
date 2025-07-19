#!/usr/bin/env python3
"""
Test script for HDF5 support in grid_utils.py
"""
import numpy as np
import h5py
import os
import sys
sys.path.append('/home/magashegyi/syncthing/Munka/Wigner/NEGF/code/python/git_test')

from utils.grid_utils import Grid, SimulationGrid

def test_grid_hdf5():
    """Test HDF5 save/load functionality for Grid class."""
    print("Testing Grid HDF5 functionality...")
    
    # Create a test grid
    original_grid = Grid(points=100, step=0.1, center=True)
    print(f"Original grid: {original_grid}")
    
    # Save to HDF5
    with h5py.File("test_grid.h5", "w") as f:
        original_grid.to_hdf5_group(f)
    
    # Load from HDF5
    with h5py.File("test_grid.h5", "r") as f:
        loaded_grid = Grid.from_hdf5_group(f)
    
    print(f"Loaded grid: {loaded_grid}")
    
    # Verify that the grids are identical
    assert np.allclose(original_grid.grid, loaded_grid.grid), "Grid arrays don't match!"
    assert original_grid.num == loaded_grid.num, "Number of points don't match!"
    assert np.isclose(original_grid.dx, loaded_grid.dx), "Step sizes don't match!"
    assert np.isclose(original_grid.width, loaded_grid.width), "Widths don't match!"
    
    print("✓ Grid HDF5 test passed!")
    
    # Clean up
    os.remove("test_grid.h5")

def test_simulation_grid_hdf5():
    """Test HDF5 save/load functionality for SimulationGrid class."""
    print("\nTesting SimulationGrid HDF5 functionality...")
    
    # Create a test simulation grid
    original_sim_grid = SimulationGrid(
        x_points=50, x_step=0.2,
        t_points=100, t_step=0.01
    )
    
    # Add some masks
    original_sim_grid.add_spatial_mask("uniform_mask", "uniform", desired_points=25)
    original_sim_grid.add_temporal_mask("range_mask", "range", start=0.2, stop=0.8)
    
    print(f"Original simulation grid: {original_sim_grid}")
    
    # Save to HDF5
    with h5py.File("test_sim_grid.h5", "w") as f:
        original_sim_grid.to_hdf5_group(f)
    
    # Load from HDF5
    with h5py.File("test_sim_grid.h5", "r") as f:
        loaded_sim_grid = SimulationGrid.from_hdf5_group(f)
    
    print(f"Loaded simulation grid: {loaded_sim_grid}")
    
    # Verify spatial and temporal grids
    assert np.allclose(original_sim_grid.spatial_grid.grid, loaded_sim_grid.spatial_grid.grid), "Spatial grids don't match!"
    assert np.allclose(original_sim_grid.temporal_grid.grid, loaded_sim_grid.temporal_grid.grid), "Temporal grids don't match!"
    
    # Verify masks
    assert "uniform_mask" in loaded_sim_grid.spatial_masks, "Spatial mask missing!"
    assert "range_mask" in loaded_sim_grid.temporal_masks, "Temporal mask missing!"
    
    assert np.array_equal(
        original_sim_grid.spatial_masks["uniform_mask"], 
        loaded_sim_grid.spatial_masks["uniform_mask"]
    ), "Spatial masks don't match!"
    
    assert np.array_equal(
        original_sim_grid.temporal_masks["range_mask"], 
        loaded_sim_grid.temporal_masks["range_mask"]
    ), "Temporal masks don't match!"
    
    print("✓ SimulationGrid HDF5 test passed!")
    
    # Clean up
    os.remove("test_sim_grid.h5")

if __name__ == "__main__":
    test_grid_hdf5()
    test_simulation_grid_hdf5()
    print("\n🎉 All HDF5 tests passed successfully!")
