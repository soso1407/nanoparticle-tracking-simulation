"""
Nanoparticle Brownian Motion Simulation (Version 3) - 3 Particle Test Version

This is a test version of the 3D simulation with only 3 particles for easier visual inspection.
All parameters except the particle count remain the same as the full simulation.
The simulation uses realistic background noise to match experimental microscopy data.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import os
import json
import pandas as pd
from tif_converter import save_frames_as_tif
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D


# Importing the 3D simulator class
from simulation_v3 import NanoparticleSimulator3D


class TestParticleSimulator3D(NanoparticleSimulator3D):
    """
    A custom 3D simulator class that directly sets the output directory.
    This bypasses the "iteration_X" directory naming convention.
    """
    
    def __init__(self, output_dir: str, **kwargs):
        """
        Initialize the test simulator with a specific output directory.
        
        Args:
            output_dir: The exact output directory to use
            **kwargs: All other parameters to pass to NanoparticleSimulator3D
        """
        # Initialize parent class
        super().__init__(**kwargs)
        
        # Override the output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_particle_paths_3d(self):
        """
        Generate a 3D plot showing the complete path of each particle.
        
        This is especially useful for the 3-particle test to visualize
        how particles move in 3D space and cross the focal plane.
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Define colormap for each particle
        colors = ['r', 'g', 'b']
        
        # Plot the path of each particle
        for i, track in enumerate(self.tracks):
            # Extract positions from track data
            if not track.positions:  # Skip if no positions
                continue
                
            # Convert positions to arrays
            x_values = []
            y_values = []
            z_values = []
            
            for pos in track.positions:
                x_values.append(pos[0])
                y_values.append(pos[1])
                z_values.append(pos[2])
            
            # Create arrays from extracted values
            x = np.array(x_values)
            y = np.array(y_values)
            z = np.array(z_values)
            
            # Create scatter plot of the path
            ax.plot(x, y, z, color=colors[i % len(colors)], alpha=0.5, label=f'Particle {i}')
            
            # Mark start and end points
            if len(x) > 0:
                ax.scatter(x[0], y[0], z[0], color=colors[i % len(colors)], marker='o', s=100)
                ax.scatter(x[-1], y[-1], z[-1], color=colors[i % len(colors)], marker='*', s=200)
        
        # Draw the focal plane
        x_range = np.linspace(0, self.frame_size[0], 10)
        y_range = np.linspace(0, self.frame_size[1], 10)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.ones_like(X) * self.focal_plane
        
        ax.plot_surface(X, Y, Z, alpha=0.2, color='gray')
        
        # Set axis labels and title
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_zlabel('Z Position (meters)')
        ax.set_title('3D Particle Paths')
        
        # Set axis limits
        ax.set_xlim(0, self.frame_size[0])
        ax.set_ylim(0, self.frame_size[1])
        ax.set_zlim(self.z_range[0], self.z_range[1])
        
        # Add a legend
        ax.legend()
        
        # Save the plot
        output_path = os.path.join(self.output_dir, "particle_3d_paths.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
    def plot_z_positions_over_time(self):
        """
        Plot the z-position of each particle over time.
        
        This visualization helps show when particles cross the focal plane
        and how that corresponds to their visibility in the 2D projection.
        """
        plt.figure(figsize=(12, 8))
        
        # Define colormap for each particle
        colors = ['r', 'g', 'b']
        
        # Plot focal plane as horizontal line
        plt.axhline(y=self.focal_plane, color='gray', linestyle='--', 
                  label='Focal Plane')
        
        # Depth of field boundaries
        plt.axhline(y=self.focal_plane + self.depth_of_field, color='gray', 
                  linestyle=':', alpha=0.5, label='Depth of Field Boundary')
        plt.axhline(y=self.focal_plane - self.depth_of_field, color='gray', 
                  linestyle=':', alpha=0.5)
        
        # Plot z-positions for each particle
        for i, track in enumerate(self.tracks):
            if not track.positions or not track.frames:  # Skip if no data
                continue
                
            # Extract z coordinates and frame numbers
            z_values = [pos[2] for pos in track.positions]
            frames = track.frames
            
            if not z_values:  # Skip if no z values
                continue
                
            # Create line plot of z-position over time
            plt.plot(frames, z_values, color=colors[i % len(colors)], label=f'Particle {i}')
            
            # Mark start and end points if we have data
            if len(frames) > 0:
                plt.scatter(frames[0], z_values[0], color=colors[i % len(colors)], marker='o', s=50)
                plt.scatter(frames[-1], z_values[-1], color=colors[i % len(colors)], marker='*', s=100)
        
        # Set axis labels and title
        plt.xlabel('Frame Number')
        plt.ylabel('Z Position (meters)')
        plt.title('Particle Z-Positions Over Time')
        
        # Set y-axis limits to the z-range with some padding
        z_min, z_max = self.z_range
        plt.ylim(z_min - 0.5e-6, z_max + 0.5e-6)
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        output_path = os.path.join(self.output_dir, "z_positions_over_time.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
    def plot_brightness_over_time(self):
        """
        Plot the brightness of each particle over time.
        
        This helps visualize how brightness changes as particles move
        through the focal plane.
        """
        plt.figure(figsize=(12, 8))
        
        # Define colormap for each particle
        colors = ['r', 'g', 'b']
        
        # Plot brightness for each particle
        for i, track in enumerate(self.tracks):
            if not track.brightnesses or not track.frames:  # Skip if no data
                continue
                
            # Extract brightness and frame numbers
            brightness = track.brightnesses
            frames = track.frames
            
            # Create line plot of brightness over time
            plt.plot(frames, brightness, color=colors[i % len(colors)], label=f'Particle {i}')
        
        # Set axis labels and title
        plt.xlabel('Frame Number')
        plt.ylabel('Normalized Brightness')
        plt.title('Particle Brightness Over Time')
        
        # Set y-axis limits
        plt.ylim(0, 1.05)
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        output_path = os.path.join(self.output_dir, "brightness_over_time.png")
        plt.savefig(output_path, dpi=300)
        plt.close()


def main():
    """Run test simulation with only 3 particles in 3D."""
    # Define the output directory name
    output_dir = os.path.join('results', 'test_3_particles_3d')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create simulator with custom test parameters
    simulator = TestParticleSimulator3D(
        output_dir=output_dir,          # Directly specify output directory
        num_particles=3,                # Only 3 particles for testing
        temperature=298.15,             # Room temperature
        viscosity=1.0e-3,               # Water viscosity
        mean_particle_radius=50e-9,     # 50 nm radius (100 nm diameter) 
        std_particle_radius=4.5e-9,     # 4.5 nm std dev (~9 nm for diameter)
        frame_size=(2048, 2048),
        pixel_size=100e-9,
        z_range=(-5e-6, 5e-6),          # -5 to 5 micrometers
        focal_plane=0.0,                # Focal plane at z=0
        depth_of_field=1e-6,            # 1 micrometer depth of field
        diffusion_time=0.01,
        gaussian_sigma=2.0,
        brightness_factor=1.0,
        background_noise=0.12,          # Increased background noise for realism
        noise_floor=50.0,               # Baseline noise floor (16-bit scale)
        noise_ceiling=2500.0,           # Maximum expected pixel value (16-bit scale)
        add_camera_noise=True,          # Add realistic camera noise
        iteration=3                     # Not used due to override
    )
    
    # Override default particle initialization to place particles at specific distances
    # This ensures we have one particle at each distance (near, medium, far)
    simulator.particle_radii = np.array([55e-9, 50e-9, 45e-9])  # Different sizes
    simulator.positions = np.array([
        [512, 1024, -0.2e-6],  # Near the focal plane 
        [1024, 512, 2.0e-6],   # Medium distance (just outside depth of field)
        [1536, 1536, 4.0e-6]   # Far from focal plane
    ])
    
    # Recalculate dependent values after overriding particle properties
    simulator.diffusion_coefficients = simulator._calculate_diffusion_coefficients()
    simulator.raw_brightnesses = simulator._calculate_raw_brightnesses()
    simulator.brightnesses = simulator._normalize_brightnesses(simulator.raw_brightnesses)
    simulator.snr_values = simulator._calculate_snr_values()
    simulator.brightness_uncertainties = simulator._calculate_brightness_uncertainties()
    
    # Plot size distribution for the 3 particles
    simulator.plot_size_distribution()
    
    # Generate and save simulation
    print(f"Generating 3D test simulation with 3 particles and focal plane effects...")
    frames = simulator.run_simulation(num_frames=100)
    simulator.save_simulation(frames, filename='simulation_v3_test_3p.gif')
    
    # Generate specialized visualizations for 3 particles
    print("Generating 3D visualization plots...")
    simulator.plot_3d_positions()
    simulator.plot_particle_paths_3d()
    simulator.plot_depth_vs_brightness()
    simulator.plot_z_positions_over_time()
    simulator.plot_brightness_over_time()
    
    print(f"3D test simulation complete. Results saved in '{output_dir}' directory.")
    
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
        
        # Print current z-position and distance from focal plane
        if i < len(simulator.tracks) and simulator.tracks[i].positions:
            last_position = simulator.tracks[i].positions[-1]
            z_pos = last_position[2]
            focal_distance = abs(z_pos - simulator.focal_plane)
            attenuation = simulator._calculate_focal_attenuation(z_pos)
            print(f"  Current z-position: {z_pos*1e6:.2f} µm")
            print(f"  Distance from focal plane: {focal_distance*1e6:.2f} µm")
            print(f"  Focal plane attenuation: {attenuation:.4f}")
        else:
            print(f"  No position data available for this particle")
        print()


if __name__ == "__main__":
    main()
    
    # Print a brief explanation of the realistic noise modifications
    print("\nNoise Characteristics:")
    print("- Background noise floor: ~50 in 16-bit scale")
    print("- Gamma-distributed noise (shape=2.0) for realistic microscopy noise")
    print("- Simulated camera effects (read noise, dark current, fixed pattern noise)")
    print("- Pixel value distribution resembling real microscopy images")
    print("- For comparison to NIST test data percentiles (54, 166, 180, etc.)")
    print("This realistic noise model produces visible but non-distracting background")
    print("that better matches experimental data for algorithm testing purposes.") 