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
                 numerical_aperture: float = 1.4,  # Added NA parameter
                 brightness_factor: float = 1.0,
                 asymmetry_factor: float = 0.1,
                 characteristic_length: Optional[float] = None,
                 particle_density: float = 1.05e3,
                 medium_density: float = 1.00e3,
                 background_noise: float = 0.12,  # Background noise level (0-1)
                 noise_floor: float = 50.0,    # Baseline noise floor (in 16-bit scale)
                 noise_ceiling: float = 2500.0, # Maximum expected pixel value (in 16-bit scale)
                 add_camera_noise: bool = True, # Whether to add realistic camera noise
                 iteration: int = 3):
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
        self.output_dir = os.path.join('results', f'iteration_{iteration}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate particle radii
        self.particle_radii = np.random.normal(
            mean_particle_radius, std_particle_radius, num_particles)
        # Ensure no negative radii
        self.particle_radii = np.abs(self.particle_radii)
        
        # Add epsilon values for numerical stability
        self.epsilon_brightness = 1e-10  # For brightness calculation
        self.epsilon_radius = 1e-9      # For radius calculation
        
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
        k = 1.0  # Scattering coefficient - could be made more precise
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
        saturation_intensity = 1e-3  # W/m² at which camera saturates
        dark_noise = 1e-9           # W/m² minimum detectable intensity
        gamma = 0.5                 # Camera gamma correction
        
        # Apply camera response curve
        normalized = np.clip((physical_intensity - dark_noise) / (saturation_intensity - dark_noise), 0, 1)
        display_value = normalized ** gamma
        
        # Apply user brightness adjustment
        return np.clip(display_value * self.brightness_factor, 0, 1)
    
    def _apply_brightness_calculations(self, particle_index: int, z_position: float) -> float:
        """Apply complete brightness calculation for a single particle."""
        # Get raw physical brightness
        physical_brightness = self.raw_brightnesses[particle_index]
        
        # Calculate attenuation from focal plane distance
        focal_attenuation = self._calculate_focal_attenuation(z_position)
        
        # Calculate final physical intensity
        final_physical_intensity = physical_brightness * focal_attenuation
        
        # Convert to display value only at the end
        return self._convert_to_display_value(final_physical_intensity)
    
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
    
    def _calculate_focal_attenuation(self, z_position: float) -> float:
        """Calculate brightness attenuation with numerical stability."""
        # Calculate distance from focal plane
        distance = z_position - self.focal_plane
        
        # Normalized distance (relative to characteristic length)
        normalized_distance = distance / self.characteristic_length
        
        # Calculate attenuation using asymmetric Lorentzian
        attenuation = 1.0 / (1.0 + normalized_distance**2 + 
                            self.asymmetry_factor * abs(normalized_distance)) + self.epsilon_brightness
        
        # Ensure minimum visibility threshold
        return max(0.05, attenuation)
    
    def _move_particles(self) -> None:
        """
        Move all particles according to Brownian motion in 3D.
        All positions and calculations are in meters.
        """
        # Calculate the step size for each particle based on diffusion coefficient
        # Standard deviation of the displacement is sqrt(2 * D * dt) in each dimension
        std_devs = np.sqrt(2 * self._calculate_diffusion_coefficient(self.mean_particle_radius) * self.diffusion_time)
        
        # Generate random displacements in all three dimensions (in meters)
        displacements = np.zeros((self.num_particles, 3))
        for i in range(3):  # x, y, z dimensions
            displacements[:, i] = np.random.normal(0, std_devs)
        
        # Update positions
        self.positions += displacements
        
        # Keep particles within frame boundaries (x and y in meters)
        frame_width_meters = self.frame_size[0] * self.pixel_size
        frame_height_meters = self.frame_size[1] * self.pixel_size
        
        for i in range(self.num_particles):
            # x boundary (horizontal)
            if self.positions[i, 0] < 0:
                self.positions[i, 0] = abs(self.positions[i, 0])
            elif self.positions[i, 0] >= frame_width_meters:
                self.positions[i, 0] = 2 * frame_width_meters - self.positions[i, 0]
            
            # y boundary (vertical)
            if self.positions[i, 1] < 0:
                self.positions[i, 1] = abs(self.positions[i, 1])
            elif self.positions[i, 1] >= frame_height_meters:
                self.positions[i, 1] = 2 * frame_height_meters - self.positions[i, 1]
            
            # z boundary (depth)
            if self.positions[i, 2] < self.z_range[0]:
                self.positions[i, 2] = 2 * self.z_range[0] - self.positions[i, 2]
            elif self.positions[i, 2] >= self.z_range[1]:
                self.positions[i, 2] = 2 * self.z_range[1] - self.positions[i, 2]
    
    def run_simulation(self, num_frames: int = 100) -> List[Image.Image]:
        """Run the simulation with continuous z-dependent blur."""
        frames = []
        
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
            
            # Create a blank frame with baseline noise
            frame = np.ones(self.frame_size, dtype=np.float32) * self.noise_floor
            
            # Add realistic background noise (more representative of microscopy data)
            if self.background_noise > 0:
                # Use gamma distribution for more realistic microscopy noise
                # (slightly skewed with longer tail than normal distribution)
                shape = 2.0  # Shape parameter for gamma distribution
                scale = self.background_noise / 2.0  # Scale parameter
                
                # Generate gamma-distributed noise
                noise = np.random.gamma(shape=shape, scale=scale, size=self.frame_size)
                
                # Add the noise to the frame
                frame += noise
                
                # Add slight spatial correlation to simulate optical system noise
                # This makes the noise pattern more natural than pure random noise
                if np.random.random() < 0.7:  # Only apply to some frames for variety
                    frame = gaussian_filter(frame, sigma=0.5)
            
            # Draw particles with z-dependent brightness
            for i in range(self.num_particles):
                # Convert physical position (meters) to pixel coordinates
                x_pixels = self.positions[i, 0] / self.pixel_size
                y_pixels = self.positions[i, 1] / self.pixel_size
                z = self.positions[i, 2]  # Keep z in meters
                
                # Calculate final brightness for this particle
                final_brightness = self._apply_brightness_calculations(i, z)
                
                # Skip rendering if brightness is too low
                if final_brightness < 0.0005:
                    continue
                
                # Calculate Gaussian sigma based on particle size and z-distance
                base_sigma = self.gaussian_sigma * (self.particle_radii[i] / self.mean_particle_radius) + self.epsilon_radius
                
                # Calculate z-dependent blur using characteristic length
                z_distance = abs(z - self.focal_plane)
                normalized_distance = z_distance / self.characteristic_length
                
                # Blur increases linearly with normalized distance, capped at maximum
                defocus_factor = 1 + min(3, normalized_distance) + self.epsilon_radius
                
                sigma = base_sigma * defocus_factor
                
                # Convert position to integer coordinates
                xi, yi = int(round(x_pixels)), int(round(y_pixels))
                
                # Define the region around the particle to draw
                # Larger sigma means we need a larger window
                window_size = int(6 * sigma) + 1
                
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
                    -((x_coords - x_pixels)**2 + (y_coords - y_pixels)**2) / (2 * sigma**2)
                )
                
                # Add the Gaussian to the frame
                frame[y_min:y_max, x_min:x_max] += gaussian
            
            # Move particles for the next frame
            self._move_particles()
            
            # Ensure values are in the expected range
            frame = np.clip(frame, 0, 1.0)
            
            # Record pixel value statistics for the first frame
            if frame_num == 0:
                self._calculate_pixel_statistics(frame)
            
            # Convert to 8-bit for GIF
            frame_8bit = (frame * 255).astype(np.uint8)
            
            # Convert to PIL Image
            pil_frame = Image.fromarray(frame_8bit)
            frames.append(pil_frame)
        
        # Print final statistics
        total_time = time.time() - start_time
        print(f"Simulation complete. Total time: {total_time:.1f} seconds")
        print(f"Average time per frame: {total_time/num_frames:.3f} seconds")
        
        return frames
    
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
            frames: List of frame images.
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
        
        # Save as TIF - keep existing code for TIF saving
        tif_filename = filename.replace('.gif', '.tif')
        tif_path = os.path.join(self.output_dir, tif_filename)
        
        # Convert frames to numpy arrays with float32 type for 16-bit TIF
        frame_arrays = [(np.array(frame) / 255.0).astype(np.float32) for frame in frames]
        
        # Scale to 16-bit range
        frame_arrays_16bit = [arr * 65535 for arr in frame_arrays]
        
        # Create 16-bit TIF images
        tif_frames = [Image.fromarray(arr.astype(np.uint16), mode='I;16') for arr in frame_arrays_16bit]
        
        # Save as multi-page TIF
        tif_frames[0].save(
            tif_path,
            save_all=True,
            append_images=tif_frames[1:],
            format='TIFF'
        )
        print(f"Saved 16-bit TIF to {tif_path}")
        
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
        colors = plt.cm.viridis(self.brightnesses)
        
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
            c=self.brightnesses[visible_indices],
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


def main():
    """Run the 3D nanoparticle simulation."""
    # Create simulator with default parameters
    simulator = NanoparticleSimulator3D(
        temperature=298.15,            # Room temperature
        viscosity=1.0e-3,              # Water viscosity
        mean_particle_radius=50e-9,    # 50 nm radius (100 nm diameter) 
        std_particle_radius=10e-9,     # 10 nm std dev (~20 nm for diameter)
        frame_size=(512, 512),         # Frame size
        pixel_size=100e-9,             # Pixel size (100 nm)
        z_range=(-10e-6, 10e-6),       # -10 to 10 micrometers in z
        focal_plane=0.0,               # Focal plane at z=0
        diffusion_time=0.1,            # Time between frames
        num_particles=100,             # Number of particles
        wavelength=550e-9,             # Default 550nm (green light)
        numerical_aperture=1.4,        # Added NA parameter
        brightness_factor=1.0,         # Brightness scaling factor
        asymmetry_factor=0.1,          # Slight asymmetry in z-attenuation
        characteristic_length=None,   # Will be set based on wavelength
        particle_density=1.05e3,        # Default: polystyrene density in kg/m³
        medium_density=1.00e3,         # Default: water density in kg/m³
        background_noise=0.12,         # Increased background noise for realism
        noise_floor=50.0,              # Baseline noise floor (in 16-bit scale)
        noise_ceiling=2500.0,          # Maximum expected pixel value (in 16-bit scale)
        add_camera_noise=True,         # Add realistic camera noise
        iteration=3                    # Iteration number for directory
    )
    
    # Plot size distribution
    simulator.plot_size_distribution()
    
    # Run simulation for 100 frames
    print("Generating 3D simulation with focal plane effects...")
    frames = simulator.run_simulation(num_frames=100)
    
    # Save the simulation
    simulator.save_simulation(frames)
    
    # Plot 3D positions and depth vs. brightness
    print("Generating visualization plots...")
    simulator.plot_3d_positions()
    simulator.plot_depth_vs_brightness()
    
    print("3D simulation complete. Results saved in 'results/iteration_3' directory.")


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
    frames = simulator.run_simulation(num_frames=num_frames)
    
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