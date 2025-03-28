"""
Nanoparticle Brownian Motion Simulation (Version 2) - 3 Particle Test Version

This is a test version of the simulation with only 3 particles for easier visual inspection.
All other parameters remain the same as the full simulation.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import os
import json
import pandas as pd
from tif_converter import save_frames_as_tif
from scipy.ndimage import gaussian_filter  # Adding this to match main simulation file


# Importing the NanoparticleSimulatorV2 class to avoid code duplication
from simulation_v2 import NanoparticleSimulatorV2


class TestParticleSimulator(NanoparticleSimulatorV2):
    """
    A custom simulator class that directly sets the output directory.
    This bypasses the "iteration_X" directory naming convention.
    """
    
    def __init__(self, output_dir: str, **kwargs):
        """
        Initialize the test simulator with a specific output directory.
        
        Args:
            output_dir: The exact output directory to use
            **kwargs: All other parameters to pass to NanoparticleSimulatorV2
        """
        # Initialize parent class
        super().__init__(**kwargs)
        
        # Override the output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)


def main():
    """Run test simulation with only 3 particles."""
    # Define the output directory name
    output_dir = os.path.join('results', 'test_3_particles_gaussian')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create simulator for room temperature with only 3 particles
    simulator = TestParticleSimulator(
        output_dir=output_dir,  # Directly specify output directory
        num_particles=3,        # Only 3 particles for testing
        temperature=298.15,     # Room temperature
        viscosity=1.0e-3,       # Water viscosity
        mean_particle_radius=50e-9,   # 50 nm radius (100 nm diameter) 
        std_particle_radius=4.5e-9,   # 4.5 nm std dev (~9 nm for diameter)
        frame_size=(2048, 2048),
        pixel_size=100e-9,
        diffusion_time=0.01,
        gaussian_sigma=2.0,
        brightness_factor=1.0,
        iteration=2  # This won't affect the output dir now
    )
    
    # Plot size distribution for the 3 particles
    simulator.plot_size_distribution()
    
    # Generate and save simulation
    print(f"Generating test simulation with 3 particles using direct Gaussian rendering...")
    frames = simulator.run_simulation(num_frames=100)
    simulator.save_simulation(frames, filename='simulation_v2_test_3p_gaussian.gif')
    
    # Plot MSD (though with only 3 particles, this may not be as meaningful)
    print("Plotting Mean Squared Displacement...")
    simulator.plot_msd(num_steps=1000)
    
    print(f"Test simulation complete. Results saved in '{output_dir}' directory.")
    
    # Print details about the 3 particles to console for quick reference
    print("\nParticle Details:")
    for i in range(3):
        print(f"Particle {i}:")
        print(f"  Size: {simulator.particle_radii[i] * 2e9:.2f} nm (diameter)")
        print(f"  Diffusion coefficient: {simulator.diffusion_coefficients[i]:.2e} m²/s")
        print(f"  Brightness: {simulator.brightnesses[i]:.4f}")
        print(f"  Raw brightness (volume-based): {simulator.raw_brightnesses[i]:.4e}")
        print(f"  SNR: {simulator.snr_values[i]:.2f}")
        print(f"  Brightness uncertainty: {simulator.brightness_uncertainties[i]:.4f}")
        print()


if __name__ == "__main__":
    main() 