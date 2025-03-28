"""
Nanoparticle Brownian Motion Simulation

This module implements a 2D Brownian motion simulation for nanoparticles using the
Stokes-Einstein equation. The simulation balances physical fidelity with computational
efficiency and is designed to be easily expandable to 3D motion.

Key Features:
- 2D Brownian motion simulation using Stokes-Einstein equation
- Gaussian blur for realistic particle visualization
- Configurable simulation parameters
- Easy expansion to 3D motion
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import os
from tif_converter import save_frames_as_tif

class NanoparticleSimulator:
    """
    Simulates nanoparticle motion using the Stokes-Einstein equation.
    
    The Stokes-Einstein equation describes the diffusion coefficient D:
    D = k_B * T / (6 * π * η * r)
    where:
    - k_B: Boltzmann constant
    - T: Temperature
    - η: Dynamic viscosity
    - r: Particle radius
    
    For 2D motion, the mean squared displacement (MSD) is:
    MSD = 4 * D * t
    """
    
    def __init__(
        self,
        num_particles: int = 100,
        temperature: float = 298.15,  # Room temperature in Kelvin
        viscosity: float = 1.0e-3,    # Water viscosity in Pa·s
        particle_radius: float = 50e-9,  # 50 nm in meters
        frame_size: Tuple[int, int] = (2048, 2048),  # Changed to 2048x2048
        pixel_size: float = 100e-9,   # 100 nm per pixel
        diffusion_time: float = 1.0,   # Time step in seconds
        gaussian_sigma: float = 2.0    # Standard deviation for particle blur
    ):
        """
        Initialize the simulator with physical parameters.
        
        Args:
            num_particles: Number of particles to simulate
            temperature: Temperature in Kelvin
            viscosity: Dynamic viscosity in Pa·s
            particle_radius: Particle radius in meters
            frame_size: Size of simulation frame in pixels
            pixel_size: Physical size of one pixel in meters
            diffusion_time: Time step for simulation in seconds
            gaussian_sigma: Standard deviation for particle blur in pixels
        """
        # Physical constants
        self.k_B = 1.380649e-23  # Boltzmann constant in J/K
        
        # Simulation parameters
        self.num_particles = num_particles
        self.temperature = temperature
        self.viscosity = viscosity
        self.particle_radius = particle_radius
        self.frame_size = frame_size
        self.pixel_size = pixel_size
        self.diffusion_time = diffusion_time
        self.gaussian_sigma = gaussian_sigma
        
        # Calculate diffusion coefficient using Stokes-Einstein equation
        self.diffusion_coefficient = (
            self.k_B * self.temperature / 
            (6 * np.pi * self.viscosity * self.particle_radius)
        )
        
        # Initialize particle positions randomly
        self.positions = np.random.rand(num_particles, 2) * frame_size
        
        # Create output directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
    
    def step(self) -> np.ndarray:
        """
        Perform one step of the simulation using the Stokes-Einstein equation.
        
        For 2D motion, the displacement follows a normal distribution with:
        mean = 0
        variance = 2 * D * dt
        
        Returns:
            Updated positions of all particles
        """
        # Calculate root mean squared displacement
        msd = 4 * self.diffusion_coefficient * self.diffusion_time
        std_dev = np.sqrt(msd) / self.pixel_size  # Convert to pixels
        
        # Generate random displacements
        displacements = np.random.normal(0, std_dev, (self.num_particles, 2))
        
        # Update positions
        self.positions += displacements
        
        # Apply periodic boundary conditions
        self.positions[:, 0] = self.positions[:, 0] % self.frame_size[0]
        self.positions[:, 1] = self.positions[:, 1] % self.frame_size[1]
        
        return self.positions
    
    def generate_frame(self) -> Image.Image:
        """
        Generate a single frame of the simulation with Gaussian-blurred particles.
        
        Returns:
            PIL Image containing the simulated frame
        """
        # Create a new image with dark background
        image = Image.new('L', self.frame_size, 0)  # Changed to black background
        draw = ImageDraw.Draw(image)
        
        # Draw particles as circles with Gaussian blur
        for pos in self.positions:
            x, y = pos
            # Draw a circle for each particle
            radius = 3  # Base radius in pixels
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=255  # Changed to white particles
            )
        
        # Apply Gaussian blur
        image = image.filter(ImageFilter.GaussianBlur(radius=self.gaussian_sigma))
        
        return image
    
    def run_simulation(self, num_frames: int = 100) -> List[Image.Image]:
        """
        Run the simulation for a specified number of frames.
        
        Args:
            num_frames: Number of frames to generate
            
        Returns:
            List of PIL Images representing the simulation frames
        """
        frames = []
        for _ in range(num_frames):
            self.step()
            frame = self.generate_frame()
            frames.append(frame)
        return frames
    
    def save_simulation(self, frames: List[Image.Image], filename: str = 'simulation.gif'):
        """
        Save the simulation frames as a GIF and TIF.
        
        Args:
            frames: List of PIL Images
            filename: Output filename
        """
        output_path = os.path.join('results', filename)
        # Ensure all frames are in 'P' (palette) mode for GIF compatibility
        frames_converted = []
        for frame in frames:
            # Convert to RGB first to ensure consistent color space
            rgb_frame = frame.convert('RGB')
            # Then convert to P mode (palette) with adaptive palette
            p_frame = rgb_frame.convert('P', palette=Image.ADAPTIVE, colors=256)
            frames_converted.append(p_frame)
        
        # Save as animated GIF
        frames_converted[0].save(
            output_path,
            save_all=True,
            append_images=frames_converted[1:],
            duration=50,  # 50ms per frame = 20fps
            loop=0,      # 0 means loop forever
            optimize=False  # Don't optimize to preserve quality
        )
        
        # Also save as TIF format (using the same base filename)
        base_filename = os.path.splitext(filename)[0]
        save_frames_as_tif(frames, f"{base_filename}.tif")
    
    def plot_msd(self, num_steps: int = 1000) -> None:
        """
        Plot the Mean Squared Displacement (MSD) to verify diffusion behavior.
        
        Args:
            num_steps: Number of steps to simulate for MSD calculation
        """
        initial_positions = self.positions.copy()
        msd_values = []
        times = []
        
        for t in range(num_steps):
            self.step()
            if t % 10 == 0:  # Record every 10th step
                displacements = self.positions - initial_positions
                msd = np.mean(np.sum(displacements**2, axis=1))
                msd_values.append(msd * (self.pixel_size**2))  # Convert to m²
                times.append(t * self.diffusion_time)
        
        # Plot MSD vs time
        plt.figure(figsize=(10, 6))
        plt.plot(times, msd_values, 'b-', label='Simulated MSD')
        plt.plot(times, 4 * self.diffusion_coefficient * np.array(times), 'r--', 
                label='Theoretical MSD')
        plt.xlabel('Time (s)')
        plt.ylabel('MSD (m²)')
        plt.title('Mean Squared Displacement vs Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join('results', 'msd_plot.png'))
        plt.close()

def main():
    """Run a sample simulation and temperature variants."""
    # Default temperature simulation (room temperature)
    simulator_room = NanoparticleSimulator(
        num_particles=100,
        temperature=298.15,  # Room temperature in Kelvin (25°C)
        viscosity=1.0e-3,
        particle_radius=50e-9,
        frame_size=(2048, 2048),
        pixel_size=100e-9,
        diffusion_time=0.01,  # 10ms time step
        gaussian_sigma=2.0
    )
    
    # Low temperature simulation
    simulator_cold = NanoparticleSimulator(
        num_particles=100,
        temperature=278.15,  # Cold temperature in Kelvin (5°C)
        viscosity=1.5e-3,    # Higher viscosity at lower temperature
        particle_radius=50e-9,
        frame_size=(2048, 2048),
        pixel_size=100e-9,
        diffusion_time=0.01,
        gaussian_sigma=2.0
    )
    
    # High temperature simulation
    simulator_hot = NanoparticleSimulator(
        num_particles=100,
        temperature=318.15,  # Hot temperature in Kelvin (45°C)
        viscosity=0.6e-3,    # Lower viscosity at higher temperature
        particle_radius=50e-9,
        frame_size=(2048, 2048),
        pixel_size=100e-9,
        diffusion_time=0.01,
        gaussian_sigma=2.0
    )
    
    # Generate frames for all simulations
    print("Generating room temperature simulation...")
    frames_room = simulator_room.run_simulation(num_frames=100)
    
    print("Generating cold temperature simulation...")
    frames_cold = simulator_cold.run_simulation(num_frames=100)
    
    print("Generating hot temperature simulation...")
    frames_hot = simulator_hot.run_simulation(num_frames=100)
    
    # Save simulations
    print("Saving simulations...")
    simulator_room.save_simulation(frames_room, filename='simulation_room_temp.gif')
    simulator_cold.save_simulation(frames_cold, filename='simulation_cold_temp.gif')
    simulator_hot.save_simulation(frames_hot, filename='simulation_hot_temp.gif')
    
    # Plot MSD for comparison
    print("Plotting Mean Squared Displacement...")
    simulator_room.plot_msd()
    
    # Create a comparison plot of the diffusion coefficients
    temperatures = [278.15, 298.15, 318.15]  # K
    viscosities = [1.5e-3, 1.0e-3, 0.6e-3]  # Pa·s
    labels = ["Cold (5°C)", "Room (25°C)", "Hot (45°C)"]
    
    # Calculate diffusion coefficients
    k_B = 1.380649e-23  # Boltzmann constant in J/K
    particle_radius = 50e-9  # 50 nm in meters
    
    diffusion_coefficients = [
        k_B * temp / (6 * np.pi * visc * particle_radius)
        for temp, visc in zip(temperatures, viscosities)
    ]
    
    # Plot diffusion coefficients
    plt.figure(figsize=(10, 6))
    plt.bar(labels, diffusion_coefficients)
    plt.ylabel('Diffusion Coefficient (m²/s)')
    plt.title('Diffusion Coefficient vs Temperature')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join('results', 'diffusion_comparison.png'))
    plt.close()
    
    print("Simulation complete. Results saved in the 'results' directory.")

if __name__ == "__main__":
    main() 