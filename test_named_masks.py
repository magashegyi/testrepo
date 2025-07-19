#!/usr/bin/env python3
"""
Tesztszkript a továbbfejlesztett SimulationGrid osztályhoz.
Bemutatja a névvel ellátott maszkokat.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.simulation_grid import SimulationGrid

def test_named_masks():
    """Test the new named mask functionality."""
    print("=== Névvel ellátott maszkokat tesztelünk ===\n")
    
    # Szimulációs grid létrehozása
    sim_grid = SimulationGrid(
        x_points=1000, x_step=0.01,  # -5 és 5 között, 0.01 lépésekkel
        t_points=500, t_step=0.1     # 0 és 50 között, 0.1 lépésekkel
    )
    
    print(f"Alapgrid létrehozva:")
    print(f"  x: {sim_grid.x_num} pont, {sim_grid.x_width:.2f} szélesség")
    print(f"  t: {sim_grid.t_num} pont, {sim_grid.t_width:.2f} szélesség")
    print(f"  x tartomány: [{sim_grid.x[0]:.2f}, {sim_grid.x[-1]:.2f}]")
    print(f"  t tartomány: [{sim_grid.t[0]:.2f}, {sim_grid.t[-1]:.2f}]")
    print()
    
    # Térbeli maszkokat adunk hozzá
    print("Térbeli maszkokat hozzáadunk:")
    
    # Interakciós régió: központi rész
    sim_grid.add_spatial_mask("interaction_region", mask_type="center", center=0.0, width=2.0)
    print(f"  'interaction_region': központi 2.0 szélesség")
    
    # Jobb oldali régió
    sim_grid.add_spatial_mask("right_region", mask_type="range", start=1.0, stop=4.0)
    print(f"  'right_region': 1.0-től 4.0-ig")
    
    # Bal oldali régió
    sim_grid.add_spatial_mask("left_region", mask_type="range", start=-4.0, stop=-1.0)
    print(f"  'left_region': -4.0-től -1.0-ig")
    
    # Ritka mintavételezés
    sim_grid.add_spatial_mask("sparse_sampling", mask_type="uniform", desired_points=100)
    print(f"  'sparse_sampling': 100 egyenletes pontok")
    
    # Véletlen mintavételezés
    sim_grid.add_spatial_mask("random_sampling", mask_type="random", desired_points=150, random_seed=42)
    print(f"  'random_sampling': 150 véletlenszerű pontok")
    print()
    
    # Időbeli maszkokat adunk hozzá
    print("Időbeli maszkokat hozzáadunk:")
    
    # Korai időszak
    sim_grid.add_temporal_mask("early_time", mask_type="range", start=0.0, stop=10.0)
    print(f"  'early_time': 0.0-től 10.0-ig")
    
    # Késői időszak
    sim_grid.add_temporal_mask("late_time", mask_type="range", start=30.0, stop=50.0)
    print(f"  'late_time': 30.0-től 50.0-ig")
    
    # Pulzus régió
    sim_grid.add_temporal_mask("pulse_region", mask_type="center", center=15.0, width=5.0)
    print(f"  'pulse_region': 15.0 körül, 5.0 szélességgel")
    
    # Ritka időbeli mintavételezés
    sim_grid.add_temporal_mask("time_sampling", mask_type="uniform", desired_points=50)
    print(f"  'time_sampling': 50 egyenletes pontok")
    print()
    
    # Maszkokat listázzuk
    print("Létrehozott maszkokat listázzuk:")
    print(f"  Térbeli maszkokat: {sim_grid.list_spatial_masks()}")
    print(f"  Időbeli maszkokat: {sim_grid.list_temporal_masks()}")
    print()
    
    # Maszkokat teszteljük
    print("Maszkokat teszteljük:")
    
    # Interaction region
    interaction_x = sim_grid.get_spatial_masked_grid("interaction_region")
    print(f"  interaction_region: {len(interaction_x)} pontok")
    print(f"    tartomány: [{interaction_x[0]:.3f}, {interaction_x[-1]:.3f}]")
    
    # Pulse region
    pulse_t = sim_grid.get_temporal_masked_grid("pulse_region")
    print(f"  pulse_region: {len(pulse_t)} pontok")
    print(f"    tartomány: [{pulse_t[0]:.3f}, {pulse_t[-1]:.3f}]")
    
    # Early time
    early_t = sim_grid.get_temporal_masked_grid("early_time")
    print(f"  early_time: {len(early_t)} pontok")
    print(f"    tartomány: [{early_t[0]:.3f}, {early_t[-1]:.3f}]")
    print()
    
    # Grid info
    print("Teljes grid információ:")
    grid_info = sim_grid.get_grid_info()
    print(f"  Térbeli grid: {grid_info['spatial']['num']} pont")
    for name, info in grid_info['spatial']['masks'].items():
        print(f"    {name}: {info['points']} pont ({info['coverage']:.1f}% lefedettség)")
    
    print(f"  Időbeli grid: {grid_info['temporal']['num']} pont")
    for name, info in grid_info['temporal']['masks'].items():
        print(f"    {name}: {info['points']} pont ({info['coverage']:.1f}% lefedettség)")
    print()
    
    # Reprezentáció
    print("SimulationGrid reprezentációja:")
    print(sim_grid)
    
    return sim_grid

def test_mask_removal():
    """Test mask removal functionality."""
    print("\n=== Maszkokat törlünk ===\n")
    
    # Hozzunk létre egy egyszerű grid-et
    sim_grid = SimulationGrid(x_points=100, x_step=0.1, t_points=50, t_step=0.2)
    
    # Adunk hozzá maszkokat
    sim_grid.add_spatial_mask("test1", mask_type="uniform", desired_points=10)
    sim_grid.add_spatial_mask("test2", mask_type="uniform", desired_points=20)
    sim_grid.add_temporal_mask("test3", mask_type="uniform", desired_points=15)
    
    print(f"Maszkokat hozzáadás után:")
    print(f"  Térbeli: {sim_grid.list_spatial_masks()}")
    print(f"  Időbeli: {sim_grid.list_temporal_masks()}")
    
    # Töröljük a test1 maszkot
    sim_grid.remove_spatial_mask("test1")
    print(f"'test1' törölve - térbeli maszkokat: {sim_grid.list_spatial_masks()}")
    
    # Töröljük a test3 maszkot
    sim_grid.remove_temporal_mask("test3")
    print(f"'test3' törölve - időbeli maszkokat: {sim_grid.list_temporal_masks()}")
    
    # Próbálunk törölni egy nem létező maszkot
    try:
        sim_grid.remove_spatial_mask("nonexistent")
    except KeyError as e:
        print(f"Helyes hibaüzenet: {e}")

def test_backward_compatibility():
    """Test backward compatibility with legacy parameters."""
    print("\n=== Visszafelé kompatibilitás tesztje ===\n")
    
    # Régi stílusú inicializálás
    sim_grid = SimulationGrid(
        x_points=100, x_step=0.1,
        t_points=50, t_step=0.2,
        spatial_mask_points=20,
        temporal_mask_points=10,
        spatial_mask_type='uniform',
        temporal_mask_type='random',
        random_seed=42
    )
    
    print(f"Régi stílusú inicializálás:")
    print(f"  spatial_mask_arr létezik: {sim_grid.spatial_mask_arr is not None}")
    print(f"  temporal_mask_arr létezik: {sim_grid.temporal_mask_arr is not None}")
    print(f"  x_masked_points: {sim_grid.x_masked_points}")
    print(f"  t_masked_points: {sim_grid.t_masked_points}")
    print(f"  Automatikusan létrehozott 'default' maszkokat: {sim_grid.list_spatial_masks()}, {sim_grid.list_temporal_masks()}")

def visualize_masks(sim_grid):
    """Visualize the masks (requires matplotlib)."""
    print("\n=== Maszkokat vizualizáljuk ===\n")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Térbeli maszkokat
        ax1 = axes[0, 0]
        x_full = sim_grid.x
        for name in sim_grid.list_spatial_masks():
            x_masked = sim_grid.get_spatial_masked_grid(name)
            ax1.plot(x_masked, np.ones_like(x_masked) * hash(name) % 10, 'o', label=name, markersize=2)
        ax1.set_xlabel('x')
        ax1.set_ylabel('Mask ID')
        ax1.set_title('Térbeli maszkokat')
        ax1.legend()
        ax1.grid(True)
        
        # Időbeli maszkokat
        ax2 = axes[0, 1]
        t_full = sim_grid.t
        for name in sim_grid.list_temporal_masks():
            t_masked = sim_grid.get_temporal_masked_grid(name)
            ax2.plot(t_masked, np.ones_like(t_masked) * hash(name) % 10, 'o', label=name, markersize=2)
        ax2.set_xlabel('t')
        ax2.set_ylabel('Mask ID')
        ax2.set_title('Időbeli maszkokat')
        ax2.legend()
        ax2.grid(True)
        
        # Térbeli maszkokat boolean array-ként
        ax3 = axes[1, 0]
        for i, name in enumerate(sim_grid.list_spatial_masks()):
            mask = sim_grid.get_spatial_mask(name)
            ax3.plot(x_full, mask.astype(int) + i * 1.1, label=name, linewidth=1)
        ax3.set_xlabel('x')
        ax3.set_ylabel('Mask (shifted)')
        ax3.set_title('Térbeli maszkokat (boolean)')
        ax3.legend()
        ax3.grid(True)
        
        # Időbeli maszkokat boolean array-ként
        ax4 = axes[1, 1]
        for i, name in enumerate(sim_grid.list_temporal_masks()):
            mask = sim_grid.get_temporal_mask(name)
            ax4.plot(t_full, mask.astype(int) + i * 1.1, label=name, linewidth=1)
        ax4.set_xlabel('t')
        ax4.set_ylabel('Mask (shifted)')
        ax4.set_title('Időbeli maszkokat (boolean)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('/home/magashegyi/syncthing/Munka/Wigner/NEGF/code/python/git_test/mask_visualization.png', dpi=150)
        print("Ábrát elmentettük: mask_visualization.png")
        plt.close()
        
    except Exception as e:
        print(f"Ábrázolási hiba: {e}")

if __name__ == "__main__":
    # Teszteket futtatjuk
    sim_grid = test_named_masks()
    test_mask_removal()
    test_backward_compatibility()
    visualize_masks(sim_grid)
    
    print("\n=== Összes teszt sikeresen lefutott! ===")
    print("\nPélda használatra:")
    print("# SimulationGrid létrehozása")
    print("sim_grid = SimulationGrid(x_points=1000, x_step=0.01, t_points=500, t_step=0.1)")
    print()
    print("# Interakciós régió hozzáadása")
    print("sim_grid.add_spatial_mask('interaction_region', mask_type='center', center=0.0, width=2.0)")
    print()
    print("# Korai időszak hozzáadása")
    print("sim_grid.add_temporal_mask('early_time', mask_type='range', start=0.0, stop=10.0)")
    print()
    print("# Maszkolt grid-ek lekérése")
    print("interaction_x = sim_grid.get_spatial_masked_grid('interaction_region')")
    print("early_t = sim_grid.get_temporal_masked_grid('early_time')")
    print()
    print("# Maszkokat listázása")
    print("print(sim_grid.list_spatial_masks())")
    print("print(sim_grid.list_temporal_masks())")
