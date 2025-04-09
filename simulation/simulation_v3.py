"""
UPDATED VERSION WITH NEW EQUATIONS:
- Added proper sigma calculation based on wavelength and NA
- Added density parameters for particles and medium
- Updated brightness equations with epsilon values
- Removed DOF parameter in favor of physically-based blur

Nanoparticle Brownian Motion Simulation (Version 3)

Enhanced features in version 3:
- 3D particle tracking and Brownian motion
- Focal plane simulation with brightness attenuation based on z-distance
- Enhanced size-dependent Gaussian rendering
- More realistic optical physics
- Realistic background noise matching experimental microscopy data
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
import time
import imageio
import tifffile
import datetime


class ParticleTrack3D:
    """Class to store information about the track of a particle in 3D."""
    
    def __init__(self, particle_id: int):
        """
        Initialize a particle track.
        
        Args:
            particle_id: The ID of the particle.
        """
        self.particle_id = particle_id
        self.frames: List[int] = []
        self.positions: List[Tuple[float, float, float]] = []  # x, y, z positions
        self.sizes: List[float] = []                          # radius in meters
        self.brightnesses: List[float] = []                   # normalized 0-1
        self.raw_brightnesses: List[float] = []               # physical value
        self.snr_values: List[float] = []                     # signal-to-noise ratio
        self.brightness_uncertainties: List[float] = []       # uncertainty
        self.focal_attenuations: List[float] = []             # attenuation due to z-position
        
    def add_position(self, frame: int, position: Tuple[float, float, float], 
                     size: float, brightness: float, raw_brightness: float,
                     snr: float, brightness_uncertainty: float,
                     focal_attenuation: float):
        """
        Add a position to the track.
        
        Args:
            frame: The frame number.
            position: The (x, y, z) position in meters.
            size: The radius of the particle in meters.
            brightness: The normalized brightness (0-1).
            raw_brightness: The physical brightness value.
            snr: The signal-to-noise ratio.
            brightness_uncertainty: The uncertainty in brightness measurement.
            focal_attenuation: The brightness attenuation due to z-position.
        """
        self.frames.append(frame)
        self.positions.append(position)
        self.sizes.append(size)
        self.brightnesses.append(brightness)
        self.raw_brightnesses.append(raw_brightness)
        self.snr_values.append(snr)
        self.brightness_uncertainties.append(brightness_uncertainty)
        self.focal_attenuations.append(focal_attenuation)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the track to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the track.
        """
        return {
            'particle_id': self.particle_id,
            'frames': self.frames,
            'positions': [(x, y, z) for x, y, z in self.positions],
            'sizes': self.sizes,
            'brightnesses': self.brightnesses,
            'raw_brightnesses': self.raw_brightnesses,
            'snr_values': self.snr_values,
            'brightness_uncertainties': self.brightness_uncertainties,
            'focal_attenuations': self.focal_attenuations
        }
        
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the track to a pandas DataFrame.
        
        Returns:
            DataFrame representation of the track.
        """
        data = {
            'frame': self.frames,
            'particle': [self.particle_id] * len(self.frames),
            'x': [pos[0] for pos in self.positions],
            'y': [pos[1] for pos in self.positions],
            'z': [pos[2] for pos in self.positions],
            'size': self.sizes,
            'mass': [4/3 * np.pi * (s**3) for s in self.sizes],  # Volume as mass
            'brightness': self.brightnesses,
            'raw_brightness': self.raw_brightnesses,
            'brightness_uncertainty': self.brightness_uncertainties,
            'snr': self.snr_values,
            'focal_attenuation': self.focal_attenuations,
            'diffusion_coefficient': [self._calculate_diffusion_coefficient(s) for s in self.sizes]
        }
        return pd.DataFrame(data)
    
    def _calculate_diffusion_coefficient(self, r: float, 
                                        temperature: float = 298.15, 
                                        viscosity: float = 1.0e-3) -> float:
        """
        Calculate the diffusion coefficient for a particle.
        
        Args:
            r: The radius of the particle in meters.
            temperature: The temperature in Kelvin.
            viscosity: The viscosity in Pa·s.
            
        Returns:
            The diffusion coefficient in m²/s.
        """
        # Boltzmann constant
        k_B = 1.380649e-23  # J/K
        
        # Stokes-Einstein equation
        D = k_B * temperature / (6 * np.pi * viscosity * r)
        
        return D


class NanoparticleSimulator3D:
    """Class to simulate the Brownian motion of nanoparticles in 3D."""
    
    def __init__(self, 
                 temperature: float = 298.15,
                 viscosity: float = 1.0e-3,
                 mean_particle_radius: float = 50e-9,
                 std_particle_radius: float = 10e-9,
                 frame_size: Tuple[int, int] = (512, 512),
                 pixel_size: float = 100e-9,
                 z_range: Tuple[float, float] = (-10e-6, 10e-6),
                 focal_plane: float = 0.0,
                 diffusion_time: float = 0.1,
                 num_particles: int = 100,
                 wavelength: float = 550e-9,
                 numerical_aperture: float = 1.4,
                 brightness_factor: float = 1.0,
                 asymmetry_factor: float = 0.1,
                 characteristic_length: Optional[float] = None,
                 particle_density: float = 1.05e3,
                 medium_density: float = 1.00e3,
                 background_noise: float = 0.12,
                 noise_floor: float = 50.0,
                 noise_ceiling: float = 2500.0,
                 add_camera_noise: bool = True,
                 iteration: int = 3,
                 run: int = 1):
        """Initialize the simulator with physically-based Gaussian sigma."""
        self.temperature = temperature
        self.viscosity = viscosity
        self.mean_particle_radius = mean_particle_radius
        self.std_particle_radius = std_particle_radius
        self.frame_size = frame_size
        self.pixel_size = pixel_size
        self.z_range = z_range
        self.focal_plane = focal_plane
        self.diffusion_time = diffusion_time
        self.num_particles = num_particles
        self.wavelength = wavelength
        self.numerical_aperture = numerical_aperture
        self.brightness_factor = brightness_factor
        self.asymmetry_factor = asymmetry_factor
        self.particle_density = particle_density
        self.medium_density = medium_density
        
        # Calculate buoyancy factor
        self.buoyancy_factor = (particle_density - medium_density) / particle_density
        
        # Calculate characteristic length based on wavelength if not provided
        if characteristic_length is None:
            # Characteristic length scales with wavelength
            self.characteristic_length = wavelength * 4
        else:
            self.characteristic_length = characteristic_length
        
        self.background_noise = background_noise
        self.noise_floor = noise_floor / 65535.0  # Convert to 0-1 range
        self.noise_ceiling = noise_ceiling / 65535.0  # Convert to 0-1 range
        self.add_camera_noise = add_camera_noise
        
        # Set up the output directory
        self.output_dir = os.path.join('results', f'iteration_{iteration}', f'run_{run}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate particle radii
        self.particle_radii = np.random.normal(
            mean_particle_radius, std_particle_radius, num_particles)
        # Ensure no negative radii
        self.particle_radii = np.abs(self.particle_radii)
        
        # Calculate raw physical brightnesses
        self.raw_brightnesses = self._calculate_raw_brightnesses()
        
        # Initialize particle positions (3D)
        self.positions = self._initialize_positions()
        
        # Initialize list to hold the track of each particle
        self.tracks = [ParticleTrack3D(i) for i in range(num_particles)]
        
        # Store pixel value statistics after the first frame is created
        self.pixel_value_stats = {
            'min': 0,
            'max': 0,
            'mean': 0,
            'percentiles': {}
        }
        
        # Calculate diffraction-limited spot size
        diffraction_limit = wavelength / (2 * numerical_aperture)
        
        # Convert diffraction limit to pixels and ensure Nyquist sampling
        self.gaussian_sigma = (diffraction_limit / pixel_size) * (2/2.355)  # 2.355 converts FWHM to sigma
        
        print(f"Physical parameters:")
        print(f"- Wavelength: {wavelength*1e9:.0f} nm")
        print(f"- Numerical aperture: {numerical_aperture}")
        print(f"- Diffraction limit: {diffraction_limit*1e9:.0f} nm")
        print(f"- Pixel size: {pixel_size*1e9:.0f} nm")
        print(f"- Base sigma: {self.gaussian_sigma:.2f} pixels")
    
    def _calculate_raw_brightnesses(self) -> np.ndarray:
        """Calculate raw brightness values based on Rayleigh scattering."""
        # Rayleigh scattering intensity proportional to r⁶/λ⁴
        # Returns intensity in physical units (W/m²)
        k = 1e5  # Increased scattering coefficient for better visibility while maintaining physics
        raw_brightnesses = k * (self.particle_radii ** 6) / (self.wavelength ** 4)
        return raw_brightnesses
    
    def _convert_to_display_value(self, physical_intensity: float) -> float:
        """
        Convert physical intensity to display value (0-1 range).
        Models a realistic camera response.
        
        Args:
            physical_intensity: Light intensity in W/m²
            
        Returns:
            Display value between 0 and 1
        """
        # Model camera response (example parameters)
        saturation_intensity = 1e-3  # Saturation threshold
        dark_noise = 1e-9          # Dark noise floor
        gamma = 0.7                 # Less aggressive gamma for more natural falloff
        
        # Apply camera response curve with minimum value
        normalized = np.clip((physical_intensity - dark_noise) / (saturation_intensity - dark_noise), 0, 1)
        display_value = normalized ** gamma
        
        # Apply user brightness adjustment with lower minimum brightness
        return np.clip(display_value * self.brightness_factor + 0.005, 0, 1)  # Lower minimum brightness
    
    def _calculate_focal_attenuation(self, z_position: float) -> float:
        """Calculate brightness attenuation based on distance from focal plane."""
        # Calculate distance from focal plane
        distance = z_position - self.focal_plane
        
        # Normalized distance (relative to characteristic length)
        normalized_distance = distance / self.characteristic_length
        
        # Calculate attenuation using modified asymmetric Lorentzian with very strong falloff
        attenuation = 1.0 / (1.0 + normalized_distance**2 +  # Stronger quadratic term
                            self.asymmetry_factor * abs(normalized_distance))  # Linear term for asymmetry
        
        # Very low minimum visibility for dramatic changes
        return max(0.05, attenuation)  # More dramatic attenuation range
    
    def _move_particles(self) -> None:
        """Move all particles according to Brownian motion in 3D."""
        # Calculate individual diffusion coefficients for each particle
        diffusion_coefficients = np.array([
            self._calculate_diffusion_coefficient(r) for r in self.particle_radii
        ])
        
        # Calculate step sizes for each particle
        std_devs = np.sqrt(2 * diffusion_coefficients * self.diffusion_time)
        
        # Generate random displacements in all three dimensions
        displacements = np.zeros((self.num_particles, 3))
        for i in range(3):  # x, y, z dimensions
            # Add slight bias towards focal plane in z direction for better visibility
            if i == 2:  # z-dimension
                bias = (self.focal_plane - self.positions[:, 2]) * 0.01
                displacements[:, i] = np.random.normal(bias, std_devs)
            else:
                displacements[:, i] = np.random.normal(0, std_devs)
        
        # Update positions
        new_positions = self.positions + displacements
        
        # Calculate boundaries in meters
        frame_width_meters = self.frame_size[0] * self.pixel_size
        frame_height_meters = self.frame_size[1] * self.pixel_size
        
        # Handle boundary conditions with reflection
        # X boundaries
        x_reflection = new_positions[:, 0] < 0
        new_positions[x_reflection, 0] = -new_positions[x_reflection, 0]  # Reflect off left wall
        
        x_reflection = new_positions[:, 0] > frame_width_meters
        new_positions[x_reflection, 0] = 2*frame_width_meters - new_positions[x_reflection, 0]  # Reflect off right wall
        
        # Y boundaries
        y_reflection = new_positions[:, 1] < 0
        new_positions[y_reflection, 1] = -new_positions[y_reflection, 1]  # Reflect off bottom wall
        
        y_reflection = new_positions[:, 1] > frame_height_meters
        new_positions[y_reflection, 1] = 2*frame_height_meters - new_positions[y_reflection, 1]  # Reflect off top wall
        
        # Z boundaries with soft reflection (already implemented)
        z_min, z_max = self.z_range
        z_reflection = new_positions[:, 2] < z_min
        new_positions[z_reflection, 2] = z_min + abs(new_positions[z_reflection, 2] - z_min) * 0.5
        
        z_reflection = new_positions[:, 2] > z_max
        new_positions[z_reflection, 2] = z_max - abs(new_positions[z_reflection, 2] - z_max) * 0.5
        
        # Update positions
        self.positions = new_positions
    
    def run_simulation(self, num_frames: int = 100) -> Tuple[List[Image.Image], np.ndarray]:
        """
        Run the simulation with continuous z-dependent blur.
        
        Returns:
            Tuple of (8-bit frames for GIF, 16-bit frames for TIF)
        """
        frames = []  # 8-bit frames for GIF
        frames_16bit = []  # 16-bit frames for TIF
        
        # Start timing
        start_time = time.time()
        
        # Iterate through frames
        for frame_num in range(num_frames):
            # Print progress
            if frame_num % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"Generating frame {frame_num}/{num_frames}...")
                if frame_num > 0:
                    frames_per_second = frame_num / elapsed_time
                    remaining_frames = num_frames - frame_num
                    remaining_time = remaining_frames / frames_per_second
                    print(f"  Progress: {frame_num/num_frames*100:.1f}% complete")
                    print(f"  Estimated remaining time: {remaining_time:.1f} seconds")
            
            # Create a black frame
            frame = np.zeros(self.frame_size, dtype=np.float32)
            
            # Draw particles with z-dependent brightness
            for i in range(self.num_particles):
                # Convert physical position (meters) to pixel coordinates
                x_pixels = self.positions[i, 0] / self.pixel_size
                y_pixels = self.positions[i, 1] / self.pixel_size
                z = self.positions[i, 2]  # Keep z in meters
                
                # Calculate physical brightness and attenuation
                physical_brightness = self.raw_brightnesses[i]
                focal_attenuation = self._calculate_focal_attenuation(z)
                final_physical_intensity = physical_brightness * focal_attenuation
                
                # Convert to display value
                final_brightness = self._convert_to_display_value(final_physical_intensity)
                
                # Add position to track with both physical and display values
                self.add_position_to_track(
                    particle_index=i,
                    frame=frame_num,
                    position=(self.positions[i, 0], self.positions[i, 1], self.positions[i, 2]),
                    brightness=final_brightness,
                    raw_brightness=final_physical_intensity,
                    focal_attenuation=focal_attenuation
                )
                
                # Skip rendering if brightness is too low
                if final_brightness < 0.0001:
                    continue
                
                # Calculate Gaussian sigma based on particle size and z-distance
                base_sigma = self.gaussian_sigma * (self.particle_radii[i] / self.mean_particle_radius) * 0.5
                
                # Calculate z-dependent blur using characteristic length
                z_distance = abs(z - self.focal_plane)
                normalized_distance = z_distance / self.characteristic_length
                
                # Blur increases linearly with normalized distance, capped at maximum
                defocus_factor = 1 + min(1.0, normalized_distance)
                sigma = base_sigma * defocus_factor
                
                # Convert position to integer coordinates
                xi, yi = int(round(x_pixels)), int(round(y_pixels))
                
                # Define the region around the particle to draw
                window_size = int(3 * sigma) + 1
                
                # Create bounds for the drawing window
                x_min = max(0, xi - window_size // 2)
                x_max = min(self.frame_size[0], xi + window_size // 2 + 1)
                y_min = max(0, yi - window_size // 2)
                y_max = min(self.frame_size[1], yi + window_size // 2 + 1)
                
                # Skip if the particle is entirely outside the frame
                if x_min >= x_max or y_min >= y_max:
                    continue
                
                # Calculate coordinates for the Gaussian
                x_coords, y_coords = np.meshgrid(
                    np.arange(x_min, x_max),
                    np.arange(y_min, y_max)
                )
                
                # Calculate the Gaussian values
                gaussian = final_brightness * np.exp(
                    -((x_coords - x_pixels)**2 + (y_coords - y_pixels)**2) / (sigma**2)
                )
                
                # Add the Gaussian to the frame
                frame[y_min:y_max, x_min:x_max] += gaussian
            
            # Move particles for the next frame
            self._move_particles()
            
            # Add camera noise if enabled
            if self.add_camera_noise:
                frame = np.maximum(frame + np.random.normal(0.0005, 0.0005, self.frame_size), 0)
            
            # Record pixel value statistics for the first frame
            if frame_num == 0:
                self._calculate_pixel_statistics(frame)
            
            # Apply contrast enhancement
            p0, p100 = np.percentile(frame, (0.01, 99.99))
            frame_rescaled = np.clip((frame - p0) / (p100 - p0), 0, 1)
            
            # Create 16-bit version (0-65535)
            frame_16bit = (frame_rescaled * 65535.0).astype(np.uint16)
            frames_16bit.append(frame_16bit)
            
            # Create 8-bit version for GIF (0-255)
            frame_8bit = (frame_rescaled * 255).astype(np.uint8)
            frames.append(Image.fromarray(frame_8bit, mode='L'))
            
            # Print value range for the first frame
            if frame_num == 0:
                print(f"Frame value range (8-bit): {frame_8bit.min()}-{frame_8bit.max()}")
                print(f"Frame value range (16-bit): {frame_16bit.min()}-{frame_16bit.max()}")
                print(f"Percentiles: 0.01%={p0:.6f}, 99.99%={p100:.6f}")
        
        # Print final statistics
        total_time = time.time() - start_time
        print(f"Simulation complete. Total time: {total_time:.1f} seconds")
        print(f"Average time per frame: {total_time/num_frames:.3f} seconds")
        
        # Convert frames_16bit to numpy array
        frames_16bit_array = np.stack(frames_16bit)
        
        return frames, frames_16bit_array
    
    def _calculate_pixel_statistics(self, frame: np.ndarray) -> None:
        """
        Calculate pixel value statistics for a frame.
        
        Args:
            frame: The frame to calculate statistics for.
        """
        # Calculate basic statistics
        self.pixel_value_stats['min'] = float(frame.min())
        self.pixel_value_stats['max'] = float(frame.max())
        self.pixel_value_stats['mean'] = float(frame.mean())
        
        # Calculate percentiles
        percentiles = [0, 25, 50, 75, 90, 95, 99, 99.999]
        for p in percentiles:
            self.pixel_value_stats['percentiles'][str(p)] = float(np.percentile(frame, p))
            
        # Print statistics for reference
        print("\nFrame Pixel Statistics (0-1 scale):")
        print(f"  Min: {self.pixel_value_stats['min']:.6f}")
        print(f"  Max: {self.pixel_value_stats['max']:.6f}")
        print(f"  Mean: {self.pixel_value_stats['mean']:.6f}")
        for p in percentiles:
            print(f"  {p}th percentile: {self.pixel_value_stats['percentiles'][str(p)]:.6f}")
        
        # Print 16-bit equivalent values
        print("\nFrame Pixel Statistics (16-bit scale, 0-65535):")
        print(f"  Min: {int(self.pixel_value_stats['min'] * 65535)}")
        print(f"  Max: {int(self.pixel_value_stats['max'] * 65535)}")
        print(f"  Mean: {self.pixel_value_stats['mean'] * 65535:.2f}")
        for p in percentiles:
            print(f"  {p}th percentile: {int(self.pixel_value_stats['percentiles'][str(p)] * 65535)}")
        print()
    
    def save_simulation(self, frames: List[Image.Image], filename: str = 'simulation_v3.gif') -> None:
        """
        Save the simulation as a GIF and TIF.
        
        Args:
            frames: List of frame images (8-bit for GIF).
            filename: The filename to save the simulation as.
        """
        # Save as GIF
        gif_path = os.path.join(self.output_dir, filename)
        
        # Convert frames to 'L' mode first, then to 'P' mode for GIF
        gif_frames = []
        for frame in frames:
            # Convert to 'L' (grayscale) if not already
            if frame.mode != 'L':
                frame = frame.convert('L')
            # Convert to 'P' (palette) with adaptive palette
            p_frame = frame.convert('P', palette=Image.ADAPTIVE, colors=256)
            gif_frames.append(p_frame)
        
        # Save the GIF with converted frames
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            optimize=False,
            duration=self.diffusion_time * 1000,  # milliseconds
            loop=0
        )
        print(f"Saved GIF to {gif_path}")
        
        # Save as 16-bit TIF
        tif_filename = filename.replace('.gif', '.tif')
        tif_path = os.path.join(self.output_dir, tif_filename)
        
        # Save using tifffile with explicit 16-bit settings
        tifffile.imwrite(
            tif_path,
            data=self.frames_16bit_array,
            photometric='minisblack',
            planarconfig='contig',
            dtype=np.uint16,
            metadata={'axes': 'TYX'}
        )
        print(f"Saved 16-bit TIF to {tif_path}")
        
        # Verify saved file
        saved_array = tifffile.imread(tif_path)
        print("\nVerifying saved TIF file:")
        print(f"Shape: {saved_array.shape}")
        print(f"Data type: {saved_array.dtype}")
        print(f"Value range: {saved_array.min()}-{saved_array.max()}")
        print(f"Mean value: {saved_array.mean():.2f}")
        
        # Save metadata
        self._save_metadata(tif_path)
        
        # Export track data
        self._export_tracks()
    
    def _save_metadata(self, tif_path: str) -> None:
        """
        Save simulation metadata to a JSON file.
        
        Args:
            tif_path: The path to the TIF file.
        """
        metadata = {
            'temperature': self.temperature,
            'viscosity': self.viscosity,
            'mean_particle_radius': self.mean_particle_radius,
            'std_particle_radius': self.std_particle_radius,
            'frame_size': self.frame_size,
            'pixel_size': self.pixel_size,
            'z_range': self.z_range,
            'focal_plane': self.focal_plane,
            'diffusion_time': self.diffusion_time,
            'num_particles': self.num_particles,
            'wavelength': self.wavelength,
            'numerical_aperture': self.numerical_aperture,
            'characteristic_length': self.characteristic_length,
            'background_noise': self.background_noise,
            'noise_floor': self.noise_floor * 65535.0,  # Convert back to 16-bit scale
            'noise_ceiling': self.noise_ceiling * 65535.0,  # Convert back to 16-bit scale
            'pixel_value_range': [
                int(self.pixel_value_stats['min'] * 65535),
                int(self.pixel_value_stats['max'] * 65535)
            ],
            'pixel_value_percentiles': {
                k: int(v * 65535) for k, v in self.pixel_value_stats['percentiles'].items()
            },
            'tif_bit_depth': 16,
            'add_camera_noise': self.add_camera_noise
        }
        
        # Save metadata to JSON
        metadata_path = tif_path.replace('.tif', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved metadata to {metadata_path}")
    
    def _export_tracks(self) -> None:
        """
        Export particle tracks to CSV and JSON files.
        """
        # Combine all tracks into a single DataFrame
        all_tracks = pd.concat([track.to_dataframe() for track in self.tracks])
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'simulation_v3_tracks.csv')
        all_tracks.to_csv(csv_path, index=False)
        print(f"Saved track data to {csv_path}")
        
        # Also save tracks as JSON for completeness
        json_path = os.path.join(self.output_dir, 'simulation_v3_tracks.json')
        with open(json_path, 'w') as f:
            json.dump([track.to_dict() for track in self.tracks], f, indent=2)
        print(f"Saved track data to {json_path}")
    
    def plot_size_distribution(self) -> None:
        """
        Plot the size distribution of the particles.
        """
        plt.figure(figsize=(10, 6))
        
        # Convert radii to diameters in nanometers
        diameters = self.particle_radii * 2e9
        
        plt.hist(diameters, bins=20, alpha=0.7, color='blue')
        plt.axvline(x=np.mean(diameters), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(diameters):.1f} nm')
        plt.axvline(x=np.median(diameters), color='green', linestyle='-', 
                    label=f'Median: {np.median(diameters):.1f} nm')
        
        plt.xlabel('Particle Diameter (nm)')
        plt.ylabel('Count')
        plt.title('Particle Size Distribution')
        plt.grid(alpha=0.3)
        plt.legend()
        
        # Save the plot
        output_path = os.path.join(self.output_dir, 'size_distribution.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Saved size distribution plot to {output_path}")
    
    def plot_3d_positions(self) -> None:
        """
        Plot the 3D positions of particles at the end of the simulation.
        Converts positions to appropriate units for visualization.
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Convert positions to appropriate units
        x_microns = self.positions[:, 0] * 1e6  # Convert to microns
        y_microns = self.positions[:, 1] * 1e6  # Convert to microns
        z_microns = self.positions[:, 2] * 1e6  # Convert to microns
        
        # Get colormap for brightness visualization
        normalized_brightnesses = self.raw_brightnesses / np.max(self.raw_brightnesses)
        colors = plt.cm.viridis(normalized_brightnesses)
        
        # Get attenuation for each particle
        attenuations = np.array([self._calculate_focal_attenuation(z) for z in self.positions[:, 2]])
        
        # Size factor for visualization (larger particles = larger markers)
        size_factor = 100
        sizes = (self.particle_radii / self.mean_particle_radius) * size_factor
        
        # Only show particles with some visibility
        visible_indices = attenuations > 0.05
        
        # Plot particles
        scatter = ax.scatter(
            x_microns[visible_indices], 
            y_microns[visible_indices], 
            z_microns[visible_indices],
            c=normalized_brightnesses[visible_indices],
            s=sizes[visible_indices],
            alpha=attenuations[visible_indices],
            cmap='viridis'
        )
        
        # Draw the focal plane
        x_range = np.linspace(0, self.frame_size[0] * self.pixel_size * 1e6, 10)  # Convert to microns
        y_range = np.linspace(0, self.frame_size[1] * self.pixel_size * 1e6, 10)  # Convert to microns
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.ones_like(X) * (self.focal_plane * 1e6)  # Convert to microns
        
        ax.plot_surface(X, Y, Z, alpha=0.2, color='gray')
        
        # Set axis labels and title
        ax.set_xlabel('X Position (µm)')
        ax.set_ylabel('Y Position (µm)')
        ax.set_zlabel('Z Position (µm)')
        ax.set_title('3D Particle Positions')
        
        # Set axis limits
        ax.set_xlim(0, self.frame_size[0] * self.pixel_size * 1e6)  # Convert to microns
        ax.set_ylim(0, self.frame_size[1] * self.pixel_size * 1e6)  # Convert to microns
        ax.set_zlim(self.z_range[0] * 1e6, self.z_range[1] * 1e6)  # Convert to microns
        
        # Add color bar for brightness
        cbar = plt.colorbar(scatter)
        cbar.set_label('Normalized Brightness')
        
        # Save the plot
        output_path = os.path.join(self.output_dir, '3d_positions.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Saved 3D positions plot to {output_path}")
    
    def plot_depth_vs_brightness(self) -> None:
        """
        Plot the relationship between depth (z-position) and brightness attenuation.
        
        This shows how particles fade as they move away from the focal plane.
        """
        plt.figure(figsize=(12, 8))
        
        # Generate z positions across the z range
        z_positions = np.linspace(self.z_range[0], self.z_range[1], 1000)
        
        # Calculate attenuation for each z position
        attenuations = np.array([self._calculate_focal_attenuation(z) for z in z_positions])
        
        # Plot attenuation curve
        plt.plot(z_positions * 1e6, attenuations, 'b-', linewidth=2)
        
        # Mark the focal plane
        plt.axvline(x=self.focal_plane * 1e6, color='r', linestyle='--', 
                  label=f'Focal Plane (z={self.focal_plane*1e6:.1f} µm)')
        
        # Add scatter points for actual particles
        z_positions_actual = [pos[2] for pos in self.positions]
        attenuations_actual = [self._calculate_focal_attenuation(z) for z in z_positions_actual]
        
        plt.scatter(np.array(z_positions_actual) * 1e6, attenuations_actual, 
                  c='r', alpha=0.5, label='Actual Particles')
        
        # Set axis labels and title
        plt.xlabel('Z Position (µm)')
        plt.ylabel('Brightness Attenuation Factor')
        plt.title('Depth vs. Brightness Attenuation')
        
        # Set y-axis limits
        plt.ylim(0, 1.05)
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        output_path = os.path.join(self.output_dir, 'depth_vs_brightness.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Saved depth vs. brightness plot to {output_path}")

    def add_position_to_track(self, particle_index: int, frame: int, position: Tuple[float, float, float],
                             brightness: float, raw_brightness: float, focal_attenuation: float):
        """Add a position to a particle's track."""
        self.tracks[particle_index].add_position(
            frame=frame,
            position=position,
            size=self.particle_radii[particle_index],
            brightness=brightness,
            raw_brightness=raw_brightness,
            snr=brightness / self.background_noise if self.background_noise > 0 else float('inf'),
            brightness_uncertainty=np.sqrt(brightness) * 0.1,  # Example uncertainty calculation
            focal_attenuation=focal_attenuation
        )

    def _initialize_positions(self) -> np.ndarray:
        """
        Initialize the positions of all particles in 3D space.
        
        Modified to create a more balanced distribution of particles across
        the z-range, ensuring particles appear at all distances from the focal plane.
        Physical positions are stored in meters, but x,y are converted from pixels.
        
        Returns:
            Array of particle positions (num_particles, 3) in meters.
        """
        # Initialize positions array (in meters)
        positions = np.zeros((self.num_particles, 3))
        
        # x, y positions in meters - uniform distribution across frame
        # Convert frame size from pixels to meters
        frame_width_meters = self.frame_size[0] * self.pixel_size
        frame_height_meters = self.frame_size[1] * self.pixel_size
        
        positions[:, 0] = np.random.uniform(0, frame_width_meters, self.num_particles)
        positions[:, 1] = np.random.uniform(0, frame_height_meters, self.num_particles)
        
        # For z-positions, create a stratified distribution to ensure coverage at all distances
        z_min, z_max = self.z_range
        z_range_length = z_max - z_min
        
        # Handle edge case: very few particles
        if self.num_particles < 10:
            # For small number of particles, just distribute them evenly
            if self.num_particles == 1:
                z_positions = np.array([self.focal_plane])  # Single particle at focal plane
            else:
                # Distribute evenly across z-range
                z_positions = np.linspace(z_min, z_max, self.num_particles)
                # Add small random offsets
                z_positions += np.random.normal(0, z_range_length * 0.05, self.num_particles)
                # Ensure we stay within bounds
                z_positions = np.clip(z_positions, z_min, z_max)
        else:
            # Divide the z-range into multiple segments
            num_strata = min(10, max(2, self.num_particles // 10))  # At least 2 strata
            strata_size = z_range_length / num_strata
            
            # Initialize z-positions array
            z_positions = np.zeros(self.num_particles)
            
            # Calculate number of particles per stratum
            particles_per_stratum = np.ones(num_strata, dtype=int) * (self.num_particles // num_strata)
            
            # Add remaining particles to random strata
            remainder = self.num_particles % num_strata
            if remainder > 0:
                random_strata = np.random.choice(num_strata, remainder, replace=False)
                particles_per_stratum[random_strata] += 1
                
            # Generate z-positions for each stratum
            particle_index = 0
            for i in range(num_strata):
                stratum_start = z_min + i * strata_size
                stratum_end = stratum_start + strata_size
                
                # Place particles in this stratum with small random offsets
                num_in_stratum = particles_per_stratum[i]
                
                # Create a mixture of uniform and beta distribution for more natural look
                if np.random.random() < 0.7:  # 70% using uniform
                    stratum_positions = np.random.uniform(
                        stratum_start, stratum_end, num_in_stratum
                    )
                else:  # 30% using beta distribution for some clustering
                    # Create a slightly skewed distribution within the stratum
                    alpha, beta = np.random.uniform(1, 3, 2)
                    random_values = np.random.beta(alpha, beta, num_in_stratum)
                    stratum_positions = stratum_start + random_values * strata_size
                
                # Assign positions to the main array
                z_positions[particle_index:particle_index+num_in_stratum] = stratum_positions
                particle_index += num_in_stratum
            
            # Shuffle to avoid any ordering artifacts
            np.random.shuffle(z_positions)
        
        # Assign z-positions
        positions[:, 2] = z_positions
        
        return positions

    def _calculate_diffusion_coefficient(self, r: float) -> float:
        """
        Calculate the diffusion coefficient for a particle.
        
        Args:
            r: The radius of the particle in meters.
            
        Returns:
            The diffusion coefficient in m²/s.
        """
        # Boltzmann constant
        k_B = 1.380649e-23  # J/K
        
        # Stokes-Einstein equation
        D = k_B * self.temperature / (6 * np.pi * self.viscosity * r)
        
        return D


def get_next_run():
    """Get the next available run number in the current iteration directory."""
    base_dir = "results"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Get the current iteration
    iteration_dirs = [d for d in os.listdir(base_dir) if d.startswith("iteration_")]
    numeric_iterations = []
    for d in iteration_dirs:
        try:
            num = int(d.split("_")[1])
            numeric_iterations.append(num)
        except (IndexError, ValueError):
            continue
    
    if not numeric_iterations:
        current_iteration = 1
    else:
        current_iteration = max(numeric_iterations)
    
    iteration_dir = os.path.join(base_dir, f"iteration_{current_iteration}")
    if not os.path.exists(iteration_dir):
        os.makedirs(iteration_dir)
    
    # Get the next run number
    run_dirs = [d for d in os.listdir(iteration_dir) if d.startswith("run_")]
    numeric_runs = []
    for d in run_dirs:
        try:
            num = int(d.split("_")[1])
            numeric_runs.append(num)
        except (IndexError, ValueError):
            continue
    
    if not numeric_runs:
        next_run = 1
    else:
        next_run = max(numeric_runs) + 1
    
    # Create the run directory
    run_dir = os.path.join(iteration_dir, f"run_{next_run}")
    os.makedirs(run_dir)
    
    return run_dir

def main():
    """Run the main simulation."""
    # Get the next available run number
    run_dir = get_next_run()
    # Extract iteration and run numbers from the path
    parts = run_dir.split('/')
    iteration = parts[-2].split('_')[1]  # Get number after 'iteration_'
    run = parts[-1].split('_')[1]        # Get number after 'run_'
    print(f"Starting simulation run {run_dir}...")
    
    # Simulation parameters
    mean_particle_radius = 50e-9  # 50 nm radius
    std_particle_radius = 10e-9   # 10 nm standard deviation
    frame_size = (512, 512)       # pixels
    num_particles = 100           # number of particles
    brightness_factor = 15000.0   # increased brightness for better visibility
    asymmetry_factor = 0.8        # stronger 3D effect
    characteristic_length = 0.25e-6  # 0.25 µm characteristic length for z-dependent brightness
    
    # Noise parameters - minimal noise
    background_noise = 0.0        # No background noise
    noise_floor = 0.0            # No noise floor
    noise_ceiling = 1.0          # Full dynamic range
    add_camera_noise = True      # Enable camera noise simulation
    
    # Create simulator instance
    sim = NanoparticleSimulator3D(
        mean_particle_radius=mean_particle_radius,
        std_particle_radius=std_particle_radius,
        frame_size=frame_size,
        num_particles=num_particles,
        brightness_factor=brightness_factor,
        asymmetry_factor=asymmetry_factor,
        characteristic_length=characteristic_length,
        background_noise=background_noise,
        noise_floor=noise_floor,
        noise_ceiling=noise_ceiling,
        add_camera_noise=add_camera_noise,
        iteration=int(iteration),
        run=int(run)
    )
    
    # Save size distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(sim.particle_radii * 1e9, bins=20)
    plt.xlabel('Particle Radius (nm)')
    plt.ylabel('Count')
    plt.title('Particle Size Distribution')
    plt.savefig(os.path.join(run_dir, 'size_distribution.png'))
    plt.close()
    
    # Run simulation
    print("Generating 3D simulation with focal plane effects...")
    frames, frames_16bit_array = sim.run_simulation(100)  # Generate 100 frames
    
    # Save outputs
    imageio.mimsave(os.path.join(run_dir, 'simulation_v3.gif'), frames)
    tifffile.imwrite(os.path.join(run_dir, 'simulation_v3.tif'), frames_16bit_array)
    
    # Save metadata
    metadata = {
        'parameters': {
            'temperature': sim.temperature,
            'viscosity': sim.viscosity,
            'mean_particle_radius': sim.mean_particle_radius,
            'std_particle_radius': sim.std_particle_radius,
            'frame_size': sim.frame_size,
            'pixel_size': sim.pixel_size,
            'z_range': sim.z_range,
            'focal_plane': sim.focal_plane,
            'diffusion_time': sim.diffusion_time,
            'num_particles': sim.num_particles,
            'wavelength': sim.wavelength,
            'numerical_aperture': sim.numerical_aperture,
            'brightness_factor': sim.brightness_factor,
            'asymmetry_factor': sim.asymmetry_factor,
            'characteristic_length': sim.characteristic_length,
            'particle_density': sim.particle_density,
            'medium_density': sim.medium_density,
            'background_noise': sim.background_noise,
            'noise_floor': sim.noise_floor,
            'noise_ceiling': sim.noise_ceiling,
            'add_camera_noise': sim.add_camera_noise
        },
        'timestamp': datetime.datetime.now().isoformat()
    }
    with open(os.path.join(run_dir, 'simulation_v3_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Save track data
    track_data = []
    for i, track in enumerate(sim.tracks):
        track_data.append({
            'particle_id': i,
            'frames': list(range(len(track.positions))),
            'positions': track.positions,
            'brightnesses': track.brightnesses,
            'raw_brightnesses': track.raw_brightnesses,
            'attenuations': track.focal_attenuations
        })
    
    # Save as JSON
    with open(os.path.join(run_dir, 'simulation_v3_tracks.json'), 'w') as f:
        json.dump(track_data, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    # Convert track data for CSV
    csv_data = []
    for track in track_data:
        for frame_idx, frame in enumerate(track['frames']):
            csv_data.append({
                'particle_id': track['particle_id'],
                'frame': frame,
                'x': track['positions'][frame_idx][0],
                'y': track['positions'][frame_idx][1],
                'z': track['positions'][frame_idx][2],
                'brightness': track['brightnesses'][frame_idx],
                'raw_brightness': track['raw_brightnesses'][frame_idx],
                'attenuation': track['attenuations'][frame_idx]
            })
    
    # Save as CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(os.path.join(run_dir, 'simulation_v3_tracks.csv'), index=False)
    
    print("Generating visualization plots...")
    
    # Plot 3D positions
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for track in track_data:
        positions = np.array(track['positions'])
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], alpha=0.5, label=f'Particle {track["particle_id"]}')
    
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_zlabel('Z (µm)')
    ax.set_title('3D Particle Trajectories')
    plt.savefig(os.path.join(run_dir, '3d_positions.png'))
    plt.close()
    
    # Plot depth vs brightness
    plt.figure(figsize=(10, 6))
    depths = []
    brightnesses = []
    for track in track_data:
        depths.extend([pos[2] for pos in track['positions']])
        brightnesses.extend(track['brightnesses'])
    
    plt.scatter(depths, brightnesses, alpha=0.5)
    plt.xlabel('Z-depth (µm)')
    plt.ylabel('Brightness')
    plt.title('Particle Brightness vs. Depth')
    plt.savefig(os.path.join(run_dir, 'depth_vs_brightness.png'))
    plt.close()
    
    print(f"3D simulation complete. Results saved in '{run_dir}' directory.")


def create_sequence_for_tracking(num_frames=30, num_particles=50, output_prefix="tracked_sequence"):
    """
    Create a sequence of frames specifically designed for tracking analysis.
    This creates a separate output directory and saves all necessary files.
    
    Args:
        num_frames: Number of frames to generate
        num_particles: Number of particles to simulate
        output_prefix: Prefix for output files
    """
    # Create output directory
    output_dir = os.path.join('results', output_prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create custom simulator
    simulator = NanoparticleSimulator3D(
        num_particles=num_particles,
        background_noise=0.12,
        noise_floor=50.0,
        noise_ceiling=2500.0,
        add_camera_noise=True
    )
    
    # Override the output directory
    simulator.output_dir = output_dir
    
    # Run simulation
    print(f"Generating tracking sequence with {num_particles} particles...")
    frames, frames_16bit_array = simulator.run_simulation(num_frames=num_frames)
    
    # Save simulation with custom filename
    simulator.save_simulation(frames, filename=f"{output_prefix}.gif")
    
    print(f"Tracking sequence complete. Results saved in '{output_dir}' directory.")
    print(f"Track data saved as '{output_prefix}_tracks.csv' and '{output_prefix}_tracks.json'")


if __name__ == "__main__":
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run 3D nanoparticle simulation with realistic noise')
    parser.add_argument('--frames', type=int, default=100, help='Number of frames to simulate')
    parser.add_argument('--particles', type=int, default=100, help='Number of particles to simulate')
    parser.add_argument('--output_prefix', type=str, default=None, help='Prefix for output files')
    parser.add_argument('--tracking_sequence', action='store_true', help='Create sequence for tracking analysis')
    args = parser.parse_args()
    
    if args.tracking_sequence:
        create_sequence_for_tracking(
            num_frames=args.frames, 
            num_particles=args.particles,
            output_prefix=args.output_prefix or "tracked_sequence"
        )
    else:
        main() 