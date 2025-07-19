#!/usr/bin/env python3
"""Test script for the refactored simulation_grid module."""

import numpy as np
from utils.simulation_grid import Grid, SimulationGrid, MaskGenerator

def test_grid_creation():
    """Test various grid creation methods."""
    print("Testing Grid class...")
    
    # Test points + step
    grid1 = Grid(points=100, step=0.1)
    print(f"Grid1 (points+step): {grid1}")
    
    # Test width + step
    grid2 = Grid(width=10.0, step=0.1)
    print(f"Grid2 (width+step): {grid2}")
    
    # Test width + points
    grid3 = Grid(width=10.0, points=100)
    print(f"Grid3 (width+points): {grid3}")
    
    # Test start + stop + points
    grid4 = Grid(start=0, stop=10, points=100)
    print(f"Grid4 (start+stop+points): {grid4}")
    
    # Test indexing
    print(f"Grid1[0]: {grid1[0]}")
    print(f"Grid1[0:5]: {grid1[0:5]}")
    print(f"len(grid1): {len(grid1)}")
    
    print("Grid tests passed!\n")

def test_mask_generation():
    """Test mask generation."""
    print("Testing MaskGenerator class...")
    
    # Test uniform mask
    uniform_mask = MaskGenerator.uniform_mask(1000, 100)
    print(f"Uniform mask: {np.count_nonzero(uniform_mask)} true values")
    
    # Test random mask
    random_mask = MaskGenerator.random_mask(1000, 100, seed=42)
    print(f"Random mask: {np.count_nonzero(random_mask)} true values")
    
    print("Mask generation tests passed!\n")

def test_simulation_grid():
    """Test SimulationGrid class."""
    print("Testing SimulationGrid class...")
    
    # Create a simulation grid
    sim_grid = SimulationGrid(
        x_points=1000, x_step=0.01,
        t_points=500, t_step=0.1,
        spatial_mask_points=100,
        temporal_mask_points=50,
        spatial_mask_type='uniform',
        temporal_mask_type='random',
        random_seed=42
    )
    
    print(f"SimulationGrid: {sim_grid}")
    print(f"Grid info: {sim_grid.get_grid_info()}")
    
    # Test properties
    print(f"x shape: {sim_grid.x.shape}")
    print(f"t shape: {sim_grid.t.shape}")
    print(f"x_masked shape: {sim_grid.x_masked.shape if sim_grid.x_masked is not None else None}")
    print(f"t_masked shape: {sim_grid.t_masked.shape if sim_grid.t_masked is not None else None}")
    
    print("SimulationGrid tests passed!\n")

def test_error_handling():
    """Test error handling."""
    print("Testing error handling...")
    
    try:
        # This should raise an error
        Grid(points=-10, step=0.1)
        print("ERROR: Should have raised ValueError for negative points")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    try:
        # This should raise an error
        Grid(step=0.1)  # Missing required parameters
        print("ERROR: Should have raised ValueError for insufficient parameters")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    print("Error handling tests passed!\n")

if __name__ == "__main__":
    test_grid_creation()
    test_mask_generation()
    test_simulation_grid()
    test_error_handling()
    print("All tests passed! 🎉")
