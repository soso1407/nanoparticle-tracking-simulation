"""
Nanoparticle Brownian Motion Simulation (Version 2)

This module extends the original simulation with:
1. Varying particle sizes based on a normal distribution
2. Track data output for compatibility with tracking tools
3. Brightness information for each particle
4. Support for TrackPy compatibility

The simulation still uses the Stokes-Einstein equation as the physical basis.
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

class ParticleTrack:
    """
    Data structure to store a particle's track information across frames.
    
    Stores position, size, brightness and optional attributes for each frame.
    """
    
    def __init__(self, particle_id: int):
        """
        Initialize a track for a specific particle.
        
        Args:
            particle_id: Unique identifier for the particle
        """
        self.particle_id = particle_id
        self.frames = []  # List of frame numbers
        self.positions = []  # List of (x,y) positions
        self.sizes = []  # List of sizes in nm
        self.brightnesses = []  # List of brightness values
        self.additional_data = {}  # For any additional attributes
    
    def add_point(self, frame: int, position: Tuple[float, float], 
                  size: float, brightness: float, **kwargs):
        """
        Add a data point for this particle at a specific frame.
        
        Args:
            frame: Frame number
            position: (x,y) position in pixels
            size: Particle size in nanometers
            brightness: Brightness value (normalized)
            **kwargs: Additional attributes to store
        """
        self.frames.append(frame)
        self.positions.append(position)
        self.sizes.append(size)
        self.brightnesses.append(brightness)
        
        # Store any additional attributes
        for key, value in kwargs.items():
            if key not in self.additional_data:
                self.additional_data[key] = []
            self.additional_data[key].append(value)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the track to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the track
        """
        result = {
            "particle_id": self.particle_id,
            "frames": self.frames,
            "positions": self.positions,
            "sizes": self.sizes,
            "brightnesses": self.brightnesses
        }
        
        # Add any additional attributes
        for key, values in self.additional_data.items():
            result[key] = values
            
        return result
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the track to a pandas DataFrame (TrackPy compatible).
        
        Returns:
            DataFrame with track information
        """
        # Create list of dictionaries for each frame
        data = []
        for i, frame in enumerate(self.frames):
            point = {
                "frame": frame,
                "particle": self.particle_id,
                "x": self.positions[i][0],
                "y": self.positions[i][1],
                "size": self.sizes[i],
                "mass": self.brightnesses[i],  # 'mass' is TrackPy's term for brightness
            }
            
            # Add any additional attributes
            for key, values in self.additional_data.items():
                if i < len(values):
                    point[key] = values[i]
            
            data.append(point)
        
        return pd.DataFrame(data)


class NanoparticleSimulatorV2:
    """
    Enhanced nanoparticle simulator with varying particle sizes and track output.
    
    This extends the original simulator with:
    - Variable particle sizes from a normal distribution
    - Tracking of particle properties per frame
    - Export of track data compatible with TrackPy
    """
    
    def __init__(
        self,
        num_particles: int = 100,
        temperature: float = 298.15,     # Room temperature in Kelvin
        viscosity: float = 1.0e-3,       # Water viscosity in Pa·s
        mean_particle_radius: float = 50e-9,  # Mean radius (50 nm in meters)
        std_particle_radius: float = 4.5e-9,  # Std dev (~9nm diameter)
        frame_size: Tuple[int, int] = (2048, 2048),
        pixel_size: float = 100e-9,      # 100 nm per pixel
        diffusion_time: float = 1.0,     # Time step in seconds
        gaussian_sigma: float = 2.0,     # Standard deviation for particle blur
        brightness_factor: float = 1.0,  # Scaling factor for brightness
        iteration: int = 2               # Simulation iteration (for output organization)
    ):
        """
        Initialize the enhanced simulator with physical parameters.
        
        Args:
            num_particles: Number of particles to simulate
            temperature: Temperature in Kelvin
            viscosity: Dynamic viscosity in Pa·s
            mean_particle_radius: Mean particle radius in meters
            std_particle_radius: Standard deviation of particle radius in meters
            frame_size: Size of simulation frame in pixels
            pixel_size: Physical size of one pixel in meters
            diffusion_time: Time step for simulation in seconds
            gaussian_sigma: Standard deviation for particle blur in pixels
            brightness_factor: Scaling factor for particle brightness
            iteration: Simulation iteration number (for output organization)
        """
        # Physical constants
        self.k_B = 1.380649e-23  # Boltzmann constant in J/K
        
        # Simulation parameters
        self.num_particles = num_particles
        self.temperature = temperature
        self.viscosity = viscosity
        self.mean_particle_radius = mean_particle_radius
        self.std_particle_radius = std_particle_radius
        self.frame_size = frame_size
        self.pixel_size = pixel_size
        self.diffusion_time = diffusion_time
        self.gaussian_sigma = gaussian_sigma
        self.brightness_factor = brightness_factor
        self.iteration = iteration
        
        # Generate particle sizes from normal distribution
        # Constraining to reasonable range (80-120 nm diameter -> 40-60 nm radius)
        self.particle_radii = np.clip(
            np.random.normal(
                mean_particle_radius, 
                std_particle_radius, 
                num_particles
            ),
            40e-9,  # Min radius: 40 nm
            60e-9   # Max radius: 60 nm
        )
        
        # Calculate individual diffusion coefficients based on particle sizes
        self.diffusion_coefficients = self._calculate_diffusion_coefficients()
        
        # Initialize particle positions randomly
        self.positions = np.random.rand(num_particles, 2) * frame_size
        
        # Calculate brightness based on particle size
        # Larger particles appear brighter (proportional to area)
        self.brightnesses = self._calculate_brightnesses()
        
        # Create track objects for each particle
        self.tracks = [ParticleTrack(i) for i in range(num_particles)]
        
        # Current frame counter
        self.current_frame = 0
        
        # Create output directories
        self.output_dir = os.path.join('results', f'iteration_{iteration}')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _calculate_diffusion_coefficients(self) -> np.ndarray:
        """
        Calculate diffusion coefficients for each particle based on its size.
        
        Returns:
            Array of diffusion coefficients
        """
        # Apply Stokes-Einstein equation individually to each particle
        return self.k_B * self.temperature / (6 * np.pi * self.viscosity * self.particle_radii)
    
    def _calculate_brightnesses(self) -> np.ndarray:
        """
        Calculate brightness values for each particle based on its size.
        
        Brightness is proportional to the particle's volume (r³), which affects
        the peak intensity of the Gaussian in the rendered image.
        
        Returns:
            Array of brightness values (normalized)
        """
        # Brightness proportional to particle volume (r³)
        # More physically accurate for scattering intensity
        raw_brightnesses = (4/3) * np.pi * (self.particle_radii ** 3)
        
        # Normalize to 0-1 range and apply brightness factor
        min_brightness = raw_brightnesses.min()
        max_brightness = raw_brightnesses.max()
        normalized = (raw_brightnesses - min_brightness) / (max_brightness - min_brightness)
        
        # Store raw brightness values for later use
        self.raw_brightnesses = raw_brightnesses
        
        # Generate signal-to-noise ratio (SNR) based on particle size
        # Larger particles typically have better SNR
        # Base SNR around 10 with variation based on size
        base_snr = 10.0
        self.snr_values = base_snr * normalized * (1 + 0.2 * np.random.randn(self.num_particles))
        self.snr_values = np.clip(self.snr_values, 1.0, 30.0)  # Reasonable SNR range
        
        # Calculate brightness uncertainty based on SNR
        # σ = brightness / SNR
        self.brightness_uncertainties = normalized * self.brightness_factor / self.snr_values
        
        return normalized * self.brightness_factor
    
    def step(self) -> np.ndarray:
        """
        Perform one step of the simulation with varying diffusion coefficients.
        
        Returns:
            Updated positions of all particles
        """
        # Calculate mean squared displacement for each particle
        msd_values = 4 * self.diffusion_coefficients * self.diffusion_time
        std_devs = np.sqrt(msd_values) / self.pixel_size  # Convert to pixels
        
        # Generate random displacements based on individual diffusion rates
        displacements = np.zeros((self.num_particles, 2))
        for i in range(self.num_particles):
            displacements[i] = np.random.normal(0, std_devs[i], 2)
        
        # Update positions
        self.positions += displacements
        
        # Apply periodic boundary conditions
        self.positions[:, 0] = self.positions[:, 0] % self.frame_size[0]
        self.positions[:, 1] = self.positions[:, 1] % self.frame_size[1]
        
        # Record track data for this frame
        self._record_tracks()
        
        # Increment frame counter
        self.current_frame += 1
        
        return self.positions
    
    def _record_tracks(self) -> None:
        """
        Record track data for each particle at the current frame.
        """
        for i in range(self.num_particles):
            # Add random fluctuation to brightness based on uncertainty
            brightness = self.brightnesses[i]
            brightness_uncertainty = self.brightness_uncertainties[i]
            fluctuated_brightness = np.random.normal(brightness, brightness_uncertainty)
            fluctuated_brightness = max(0.01, min(1.0, fluctuated_brightness))  # Keep in reasonable range
            
            self.tracks[i].add_point(
                frame=self.current_frame,
                position=(self.positions[i, 0], self.positions[i, 1]),
                size=self.particle_radii[i] * 2e9,  # Convert to nanometers
                brightness=fluctuated_brightness,
                raw_brightness=self.raw_brightnesses[i],
                brightness_uncertainty=self.brightness_uncertainties[i],
                snr=self.snr_values[i],
                diffusion_coefficient=self.diffusion_coefficients[i]
            )
    
    def generate_frame(self) -> Image.Image:
        """
        Generate a single frame of the simulation with Gaussian particles.
        
        Particles are rendered directly as 2D Gaussians with peak intensity
        proportional to their brightness values.
        
        Returns:
            PIL Image containing the simulated frame
        """
        # Create a numpy array for the frame
        frame_array = np.zeros(self.frame_size, dtype=np.float32)
        
        # Calculate diffraction-limited sigma in pixels
        # For typical optical microscopy, sigma is related to wavelength and numerical aperture
        # We'll use the gaussian_sigma parameter as the baseline for a 100nm particle
        base_sigma = self.gaussian_sigma
        
        # Render each particle as a 2D Gaussian
        for i, pos in enumerate(self.positions):
            x, y = pos
            
            # Convert positions to integer coordinates
            x_int, y_int = int(round(x)), int(round(y))
            
            # Skip if position is outside the frame
            if (x_int < 0 or x_int >= self.frame_size[0] or 
                y_int < 0 or y_int >= self.frame_size[1]):
                continue
            
            # Scale sigma based on particle size
            # Larger particles appear slightly larger in the image
            particle_sigma = base_sigma * (self.particle_radii[i] / 50e-9) ** 0.5
            
            # Get brightness value (0-1 range)
            brightness = self.brightnesses[i]
            
            # Create a small Gaussian kernel for this particle
            # We'll use a kernel size of 6*sigma to capture most of the Gaussian
            kernel_size = max(3, int(6 * particle_sigma))
            if kernel_size % 2 == 0:  # Ensure odd kernel size
                kernel_size += 1
                
            # Define the grid for the Gaussian
            half_size = kernel_size // 2
            y_grid, x_grid = np.ogrid[-half_size:half_size+1, -half_size:half_size+1]
            
            # Calculate the 2D Gaussian
            gaussian = brightness * np.exp(-(x_grid**2 + y_grid**2) / (2 * particle_sigma**2))
            
            # Add the Gaussian to the frame at the particle position
            # Handle edge cases where the particle is near the frame boundary
            frame_y_min = max(0, y_int - half_size)
            frame_y_max = min(self.frame_size[1], y_int + half_size + 1)
            frame_x_min = max(0, x_int - half_size)
            frame_x_max = min(self.frame_size[0], x_int + half_size + 1)
            
            kernel_y_min = max(0, half_size - y_int)
            kernel_y_max = kernel_size - max(0, (y_int + half_size + 1) - self.frame_size[1])
            kernel_x_min = max(0, half_size - x_int)
            kernel_x_max = kernel_size - max(0, (x_int + half_size + 1) - self.frame_size[0])
            
            frame_array[frame_y_min:frame_y_max, frame_x_min:frame_x_max] += gaussian[kernel_y_min:kernel_y_max, kernel_x_min:kernel_x_max]
        
        # Add background noise
        background_noise = np.random.normal(0, 0.01, self.frame_size)
        frame_array += background_noise
        
        # Clip values to [0, 1] range
        frame_array = np.clip(frame_array, 0, 1)
        
        # Scale to 0-255 for PIL Image
        frame_array = (frame_array * 255).astype(np.uint8)
        
        # Create PIL Image
        image = Image.fromarray(frame_array, mode='L')
        
        return image
    
    def run_simulation(self, num_frames: int = 100) -> List[Image.Image]:
        """
        Run the simulation for a specified number of frames.
        
        Args:
            num_frames: Number of frames to generate
            
        Returns:
            List of PIL Images representing the simulation frames
        """
        # Reset frame counter
        self.current_frame = 0
        
        # Reset tracks
        self.tracks = [ParticleTrack(i) for i in range(self.num_particles)]
        
        # Generate frames
        frames = []
        for _ in range(num_frames):
            self.step()
            frame = self.generate_frame()
            frames.append(frame)
        
        return frames
    
    def save_simulation(self, frames: List[Image.Image], filename: str = 'simulation.gif'):
        """
        Save the simulation frames as GIF and 16-bit TIF, with track data in JSON and CSV.
        
        Args:
            frames: List of PIL Images
            filename: Output filename
        """
        # Base path for output files
        base_filename = os.path.splitext(filename)[0]
        
        # Save GIF animation
        output_path = os.path.join(self.output_dir, filename)
        
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
        
        # Save as 16-bit TIF format
        tif_path = os.path.join(self.output_dir, f"{base_filename}.tif")
        
        # Convert frames to 16-bit format
        frames_16bit = []
        for frame in frames:
            # Convert to grayscale if not already
            if frame.mode != 'L':
                gray_frame = frame.convert('L')
            else:
                gray_frame = frame
            
            # Convert to 16-bit ('I;16')
            # Scale values from 0-255 to 0-65535 for 16-bit range
            array = np.array(gray_frame, dtype=np.uint16) * 257  # Scale to full 16-bit range
            frame_16bit = Image.fromarray(array, mode='I;16')
            
            frames_16bit.append(frame_16bit)
        
        # Save as multi-page 16-bit TIFF
        frames_16bit[0].save(
            tif_path,
            format='TIFF',
            compression='tiff_deflate',
            save_all=True,
            append_images=frames_16bit[1:] if len(frames_16bit) > 1 else []
        )
        print(f"Saved frames as 16-bit TIF: {tif_path}")
        
        # Save track data as JSON
        json_path = os.path.join(self.output_dir, f"{base_filename}_tracks.json")
        with open(json_path, 'w') as f:
            json.dump([track.to_dict() for track in self.tracks], f, indent=2)
        
        # Save track data as CSV (TrackPy compatible)
        csv_path = os.path.join(self.output_dir, f"{base_filename}_tracks.csv")
        self.tracks_to_dataframe().to_csv(csv_path, index=False)
        
        # Save particle metadata
        metadata_path = os.path.join(self.output_dir, f"{base_filename}_metadata.json")
        metadata = {
            "temperature": self.temperature,
            "viscosity": self.viscosity,
            "mean_particle_radius_nm": self.mean_particle_radius * 1e9,
            "std_particle_radius_nm": self.std_particle_radius * 1e9,
            "frame_size": self.frame_size,
            "pixel_size_nm": self.pixel_size * 1e9,
            "diffusion_time": self.diffusion_time,
            "num_particles": self.num_particles,
            "num_frames": self.current_frame,
            "tif_bit_depth": 16  # Add information about TIF bit depth
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def tracks_to_dataframe(self) -> pd.DataFrame:
        """
        Convert all tracks to a single TrackPy-compatible DataFrame.
        
        Returns:
            DataFrame containing all particle tracks
        """
        # Convert each track to a DataFrame and concatenate
        dfs = [track.to_dataframe() for track in self.tracks]
        return pd.concat(dfs, ignore_index=True)
    
    def plot_size_distribution(self) -> None:
        """
        Plot the distribution of particle sizes.
        """
        # Convert radii to diameters in nanometers
        diameters = self.particle_radii * 2e9
        
        plt.figure(figsize=(10, 6))
        plt.hist(diameters, bins=20, edgecolor='black')
        plt.xlabel('Particle Diameter (nm)')
        plt.ylabel('Frequency')
        plt.title('Particle Size Distribution')
        plt.axvline(100, color='red', linestyle='--', label='Mean (100 nm)')
        plt.axvline(80, color='orange', linestyle='--', label='Min (80 nm)')
        plt.axvline(120, color='orange', linestyle='--', label='Max (120 nm)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        output_path = os.path.join(self.output_dir, "particle_size_distribution.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

    def plot_msd(self, num_steps: int = 1000) -> None:
        """
        Plot the Mean Squared Displacement (MSD) for different particle sizes.
        
        Args:
            num_steps: Number of steps to simulate for MSD calculation
        """
        # Group particles into small, medium, and large categories
        size_thresholds = [45e-9, 55e-9]  # Thresholds for classification (in meters)
        
        small_indices = np.where(self.particle_radii < size_thresholds[0])[0]
        large_indices = np.where(self.particle_radii > size_thresholds[1])[0]
        medium_indices = np.where(
            (self.particle_radii >= size_thresholds[0]) & 
            (self.particle_radii <= size_thresholds[1])
        )[0]
        
        # Create arrays to store positions for each size category
        initial_positions = self.positions.copy()
        
        # Store MSD values for each category
        msd_small = []
        msd_medium = []
        msd_large = []
        times = []
        
        # Run simulation and calculate MSD
        for t in range(num_steps):
            self.step()
            
            if t % 10 == 0:  # Record every 10th step
                displacements = self.positions - initial_positions
                squared_displacements = np.sum(displacements**2, axis=1)
                
                # Calculate MSD for each size category
                if len(small_indices) > 0:
                    msd_small.append(np.mean(squared_displacements[small_indices]) * (self.pixel_size**2))
                if len(medium_indices) > 0:
                    msd_medium.append(np.mean(squared_displacements[medium_indices]) * (self.pixel_size**2))
                if len(large_indices) > 0:
                    msd_large.append(np.mean(squared_displacements[large_indices]) * (self.pixel_size**2))
                
                times.append(t * self.diffusion_time)
        
        # Plot MSD vs time for each size category
        plt.figure(figsize=(12, 8))
        
        if len(small_indices) > 0:
            plt.plot(times, msd_small, 'r-', label='Small particles (d < 90 nm)')
        if len(medium_indices) > 0:
            plt.plot(times, msd_medium, 'g-', label='Medium particles (90-110 nm)')
        if len(large_indices) > 0:
            plt.plot(times, msd_large, 'b-', label='Large particles (d > 110 nm)')
        
        # Calculate theoretical MSD for average particle size
        avg_diffusion = np.mean(self.diffusion_coefficients)
        plt.plot(times, 4 * avg_diffusion * np.array(times), 'k--', label='Theoretical (average size)')
        
        plt.xlabel('Time (s)')
        plt.ylabel('MSD (m²)')
        plt.title('Mean Squared Displacement vs Time by Particle Size')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        output_path = os.path.join(self.output_dir, "msd_by_size.png")
        plt.savefig(output_path, dpi=300)
        plt.close()


def main():
    """Run enhanced simulations with varying particle sizes."""
    # Create simulator for room temperature
    simulator = NanoparticleSimulatorV2(
        num_particles=100,
        temperature=298.15,  # Room temperature
        viscosity=1.0e-3,    # Water viscosity
        mean_particle_radius=50e-9,   # 50 nm radius (100 nm diameter) 
        std_particle_radius=4.5e-9,   # 4.5 nm std dev (~9 nm for diameter)
        frame_size=(2048, 2048),
        pixel_size=100e-9,
        diffusion_time=0.01,
        gaussian_sigma=2.0,
        brightness_factor=1.0,
        iteration=2  # Second iteration of the simulation
    )
    
    # Plot size distribution
    simulator.plot_size_distribution()
    
    # Generate and save simulation
    print("Generating simulation with varying particle sizes...")
    frames = simulator.run_simulation(num_frames=100)
    simulator.save_simulation(frames, filename='simulation_v2.gif')
    
    # Plot MSD for different particle sizes
    print("Plotting Mean Squared Displacement for different particle sizes...")
    simulator.plot_msd(num_steps=1000)
    
    print("Enhanced simulation complete. Results saved in the 'results/iteration_2' directory.")


if __name__ == "__main__":
    main() 