"""
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
                 z_range: Tuple[float, float] = (-10e-6, 10e-6),  # z range in meters
                 focal_plane: float = 0.0,                        # z position of focal plane
                 depth_of_field: float = 2e-6,                    # depth of field in meters
                 diffusion_time: float = 0.1,
                 num_particles: int = 100,
                 gaussian_sigma: float = 2.0,
                 brightness_factor: float = 1.0,
                 snr_base: float = 5.0,        # Base SNR for smallest particle
                 snr_scaling: float = 2.0,     # How SNR scales with radius
                 background_noise: float = 0.12,  # Background noise level (0-1)
                 noise_floor: float = 50.0,    # Baseline noise floor (in 16-bit scale)
                 noise_ceiling: float = 2500.0, # Maximum expected pixel value (in 16-bit scale)
                 add_camera_noise: bool = True, # Whether to add realistic camera noise
                 iteration: int = 3):
        """
        Initialize the simulator.
        
        Args:
            temperature: The temperature in Kelvin.
            viscosity: The viscosity in Pa·s.
            mean_particle_radius: The mean radius of the particles in meters.
            std_particle_radius: The standard deviation of the particle radius in meters.
            frame_size: The size of the frame in pixels (width, height).
            pixel_size: The size of a pixel in meters.
            z_range: The range of z values in meters (min, max).
            focal_plane: The z position of the focal plane in meters.
            depth_of_field: The depth of field in meters (total depth).
            diffusion_time: The time between frames in seconds.
            num_particles: The number of particles to simulate.
            gaussian_sigma: The sigma parameter for the Gaussian kernel.
            brightness_factor: A scaling factor for particle brightness.
            snr_base: Base signal-to-noise ratio for smallest particle.
            snr_scaling: How SNR scales with particle radius.
            background_noise: Level of background noise (0-1).
            noise_floor: Minimum pixel value for background (0-65535).
            noise_ceiling: Maximum expected pixel value (0-65535).
            add_camera_noise: Whether to add realistic camera noise.
            iteration: Iteration number for output directory naming.
        """
        self.temperature = temperature
        self.viscosity = viscosity
        self.mean_particle_radius = mean_particle_radius
        self.std_particle_radius = std_particle_radius
        self.frame_size = frame_size
        self.pixel_size = pixel_size
        self.z_range = z_range
        self.focal_plane = focal_plane
        self.depth_of_field = depth_of_field
        self.diffusion_time = diffusion_time
        self.num_particles = num_particles
        self.gaussian_sigma = gaussian_sigma
        self.brightness_factor = brightness_factor
        self.snr_base = snr_base
        self.snr_scaling = snr_scaling
        self.background_noise = background_noise
        self.noise_floor = noise_floor / 65535.0  # Convert to 0-1 range
        self.noise_ceiling = noise_ceiling / 65535.0  # Convert to 0-1 range
        self.add_camera_noise = add_camera_noise
        
        # Set up the output directory
        self.output_dir = os.path.join('results', f'iteration_{iteration}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Boltzmann constant
        self.k_B = 1.380649e-23  # J/K
        
        # Generate particle radii
        self.particle_radii = np.random.normal(
            mean_particle_radius, std_particle_radius, num_particles)
        # Ensure no negative radii
        self.particle_radii = np.abs(self.particle_radii)
        
        # Calculate diffusion coefficients using Stokes-Einstein equation
        self.diffusion_coefficients = self._calculate_diffusion_coefficients()
        
        # Calculate brightness values
        self.raw_brightnesses = self._calculate_raw_brightnesses()
        self.brightnesses = self._normalize_brightnesses(self.raw_brightnesses)
        
        # Calculate signal-to-noise ratio for each particle based on size
        self.snr_values = self._calculate_snr_values()
        
        # Calculate brightness uncertainty based on SNR
        self.brightness_uncertainties = self._calculate_brightness_uncertainties()
        
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
    
    def _calculate_diffusion_coefficients(self) -> np.ndarray:
        """
        Calculate the diffusion coefficients for all particles.
        
        Returns:
            Array of diffusion coefficients in m²/s.
        """
        # Stokes-Einstein equation: D = k_B * T / (6 * pi * eta * r)
        return self.k_B * self.temperature / (6 * np.pi * self.viscosity * self.particle_radii)
    
    def _calculate_raw_brightnesses(self) -> np.ndarray:
        """
        Calculate the raw brightness values based on particle volume.
        
        In this version, brightness scales with volume (r³) to better reflect
        light scattering physics for nanoparticles.
        
        Returns:
            Array of raw brightness values.
        """
        # Brightness proportional to volume (r³)
        return 4/3 * np.pi * (self.particle_radii ** 3)
    
    def _normalize_brightnesses(self, raw_brightnesses: np.ndarray) -> np.ndarray:
        """
        Normalize the brightness values to the range [0, 1].
        
        Args:
            raw_brightnesses: Array of raw brightness values.
            
        Returns:
            Array of normalized brightness values.
        """
        if raw_brightnesses.max() == raw_brightnesses.min():
            return np.ones_like(raw_brightnesses)
        
        # Linear normalization
        normalized = (raw_brightnesses - raw_brightnesses.min()) / (raw_brightnesses.max() - raw_brightnesses.min())
        
        # Apply brightness factor
        return normalized * self.brightness_factor
    
    def _calculate_snr_values(self) -> np.ndarray:
        """
        Calculate the Signal-to-Noise Ratio (SNR) for each particle.
        
        The SNR increases with particle size, as larger particles scatter
        more light and produce a stronger signal.
        
        Returns:
            Array of SNR values.
        """
        # Normalize radii for scaling
        normalized_radii = self.particle_radii / self.particle_radii.min()
        
        # Calculate SNR with power law scaling
        snr = self.snr_base * (normalized_radii ** self.snr_scaling)
        
        # Ensure minimum SNR of 1.0
        snr = np.maximum(snr, 1.0)
        
        return snr
    
    def _calculate_brightness_uncertainties(self) -> np.ndarray:
        """
        Calculate brightness uncertainty based on SNR.
        
        Uncertainty is inversely proportional to SNR:
        uncertainty = brightness / SNR
        
        Returns:
            Array of brightness uncertainties.
        """
        return self.brightnesses / self.snr_values
    
    def _initialize_positions(self) -> np.ndarray:
        """
        Initialize the positions of all particles in 3D space.
        
        Modified to create a more balanced distribution of particles across
        the z-range, ensuring particles appear at all distances from the focal plane.
        
        Returns:
            Array of particle positions (num_particles, 3).
        """
        # Random positions within the frame
        positions = np.zeros((self.num_particles, 3))
        
        # x, y positions in pixels - uniform distribution
        positions[:, 0] = np.random.uniform(0, self.frame_size[0], self.num_particles)
        positions[:, 1] = np.random.uniform(0, self.frame_size[1], self.num_particles)
        
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
        """
        Calculate the brightness attenuation due to distance from the focal plane.
        
        Uses a modified Gaussian function centered at the focal plane with the depth of field
        determining the spread. Modified to make particles at medium distances more visible.
        
        Args:
            z_position: Z position of the particle in meters.
            
        Returns:
            Attenuation factor between 0 and 1.
        """
        # Distance from focal plane
        distance = abs(z_position - self.focal_plane)
        
        # Modified Gaussian attenuation with depth of field as sigma
        # Increased sigma and added a minimum attenuation floor
        
        # Normalize distance to depth of field
        normalized_distance = distance / (self.depth_of_field/2)
        
        # Core attenuation calculation - modified Gaussian with longer falloff
        if normalized_distance <= 1:
            # Within depth of field - use high attenuation (80-100%)
            attenuation = 1.0 - 0.2 * (normalized_distance ** 2)
        elif normalized_distance <= 3:
            # Medium distance - gentler falloff (20-80%)
            attenuation = 0.8 * np.exp(-0.3 * ((normalized_distance - 1) ** 2))
        else:
            # Far distance - exponential falloff with minimum floor
            attenuation = max(0.05, 0.2 * np.exp(-0.2 * (normalized_distance - 3)))
        
        return attenuation
    
    def _move_particles(self) -> None:
        """
        Move all particles according to Brownian motion in 3D.
        """
        # Calculate the step size for each particle based on diffusion coefficient
        # Standard deviation of the displacement is sqrt(2 * D * dt) in each dimension
        std_devs = np.sqrt(2 * self.diffusion_coefficients * self.diffusion_time)
        
        # Generate random displacements in all three dimensions
        displacements = np.zeros((self.num_particles, 3))
        for i in range(3):  # x, y, z dimensions
            displacements[:, i] = np.random.normal(0, std_devs)
        
        # Update positions
        self.positions += displacements
        
        # Keep particles within frame boundaries (x and y)
        for i in range(self.num_particles):
            # x boundary (horizontal)
            if self.positions[i, 0] < 0:
                self.positions[i, 0] = abs(self.positions[i, 0])
            elif self.positions[i, 0] >= self.frame_size[0]:
                self.positions[i, 0] = 2 * self.frame_size[0] - self.positions[i, 0]
            
            # y boundary (vertical)
            if self.positions[i, 1] < 0:
                self.positions[i, 1] = abs(self.positions[i, 1])
            elif self.positions[i, 1] >= self.frame_size[1]:
                self.positions[i, 1] = 2 * self.frame_size[1] - self.positions[i, 1]
            
            # z boundary (depth)
            if self.positions[i, 2] < self.z_range[0]:
                self.positions[i, 2] = 2 * self.z_range[0] - self.positions[i, 2]
            elif self.positions[i, 2] >= self.z_range[1]:
                self.positions[i, 2] = 2 * self.z_range[1] - self.positions[i, 2]
    
    def _apply_random_brightness_fluctuation(self, brightness: float, uncertainty: float) -> float:
        """
        Apply random fluctuation to brightness based on uncertainty.
        
        Args:
            brightness: Base brightness value.
            uncertainty: Uncertainty in brightness measurement.
            
        Returns:
            Fluctuated brightness value.
        """
        # Apply random fluctuation based on normal distribution
        fluctuation = np.random.normal(0, uncertainty)
        fluctuated_brightness = brightness + fluctuation
        
        # Ensure brightness is non-negative
        return max(0, fluctuated_brightness)
    
    def run_simulation(self, num_frames: int = 100) -> List[Image.Image]:
        """
        Run the simulation for the specified number of frames.
        
        Args:
            num_frames: The number of frames to simulate.
            
        Returns:
            List of frame images.
        """
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
            # Use noise floor as base level instead of zero
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
                # Extract particle position (x, y are in pixels, z in meters)
                x, y, z = self.positions[i]
                
                # Calculate attenuation based on z-position
                focal_attenuation = self._calculate_focal_attenuation(z)
                
                # Only render particles that are not completely attenuated
                # Lower threshold to see more particles at medium distances
                if focal_attenuation > 0.005:  # Lower threshold from previous 0.01
                    # Get base brightness and apply focal attenuation
                    base_brightness = self.brightnesses[i]
                    attenuated_brightness = base_brightness * focal_attenuation
                    
                    # Apply random fluctuation based on uncertainty
                    uncertainty = self.brightness_uncertainties[i]
                    fluctuated_brightness = self._apply_random_brightness_fluctuation(
                        attenuated_brightness, uncertainty)
                    
                    # Skip rendering if brightness is too low (optimization)
                    # Lower brightness threshold too
                    if fluctuated_brightness < 0.0005:  # Lower from previous 0.001
                        continue
                    
                    # Calculate Gaussian sigma based on particle size and z-distance
                    # Larger particles have wider Gaussians
                    # Particles away from focal plane have wider (more blurred) Gaussians
                    base_sigma = self.gaussian_sigma * (self.particle_radii[i] / self.mean_particle_radius)
                    
                    # Add defocus blur based on distance from focal plane
                    # Modified to create more natural blur progression
                    z_distance = abs(z - self.focal_plane)
                    
                    # Adjusted defocus factor - gentler progression for realistic appearance
                    if z_distance <= self.depth_of_field/2:
                        # Within depth of field - minimal blur increase 
                        defocus_factor = 1 + (z_distance / self.depth_of_field) * 0.5
                    else:
                        # Outside depth of field - steeper blur increase with distance
                        relative_distance = (z_distance - self.depth_of_field/2) / self.depth_of_field
                        defocus_factor = 1.25 + min(3, relative_distance * 2)
                    
                    sigma = base_sigma * defocus_factor
                    
                    # Convert position to integer coordinates
                    xi, yi = int(round(x)), int(round(y))
                    
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
                    gaussian = fluctuated_brightness * np.exp(
                        -((x_coords - x)**2 + (y_coords - y)**2) / (2 * sigma**2)
                    )
                    
                    # Add the Gaussian to the frame
                    frame[y_min:y_max, x_min:x_max] += gaussian
                
                # Record the particle's position and attributes for this frame
                raw_brightness = self.raw_brightnesses[i]
                snr = self.snr_values[i]
                brightness_uncertainty = self.brightness_uncertainties[i]
                
                self.tracks[i].add_position(
                    frame=frame_num,
                    position=(x, y, z),
                    size=self.particle_radii[i],
                    brightness=self.brightnesses[i],
                    raw_brightness=raw_brightness,
                    snr=snr,
                    brightness_uncertainty=brightness_uncertainty,
                    focal_attenuation=focal_attenuation
                )
            
            # Add camera noise effects if enabled
            if self.add_camera_noise:
                # Add read noise (multiplicative)
                read_noise = np.random.normal(1, 0.01, self.frame_size)  # 1% variation
                frame = frame * read_noise
                
                # Add dark current noise (additive Poisson noise)
                dark_current = np.random.poisson(0.5, self.frame_size) / 65535.0
                frame += dark_current
                
                # Add fixed pattern noise (spatially correlated)
                if frame_num == 0:  # Only generate the pattern once for consistency
                    self.fixed_pattern = np.random.normal(1, 0.005, self.frame_size)  # 0.5% variation
                frame = frame * self.fixed_pattern
            
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
            'depth_of_field': self.depth_of_field,
            'diffusion_time': self.diffusion_time,
            'num_particles': self.num_particles,
            'gaussian_sigma': self.gaussian_sigma,
            'brightness_factor': self.brightness_factor,
            'snr_base': self.snr_base,
            'snr_scaling': self.snr_scaling,
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
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get colormap for brightness visualization
        colors = plt.cm.viridis(self.brightnesses)
        
        # Get attenuation for each particle
        attenuations = np.array([self._calculate_focal_attenuation(z) for _, _, z in self.positions])
        
        # Size factor for visualization (larger particles = larger markers)
        size_factor = 100
        sizes = (self.particle_radii / self.mean_particle_radius) * size_factor
        
        # Only show particles with some visibility
        visible_indices = attenuations > 0.05
        
        # Plot particles
        scatter = ax.scatter(
            self.positions[visible_indices, 0], 
            self.positions[visible_indices, 1], 
            self.positions[visible_indices, 2],
            c=self.brightnesses[visible_indices],
            s=sizes[visible_indices],
            alpha=attenuations[visible_indices],
            cmap='viridis'
        )
        
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
        ax.set_title('3D Particle Positions')
        
        # Set axis limits
        ax.set_xlim(0, self.frame_size[0])
        ax.set_ylim(0, self.frame_size[1])
        ax.set_zlim(self.z_range[0], self.z_range[1])
        
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
        
        # Mark the depth of field boundaries
        plt.axvline(x=(self.focal_plane + self.depth_of_field/2) * 1e6, color='g', linestyle=':', 
                  label=f'Depth of Field (±{self.depth_of_field/2*1e6:.1f} µm)')
        plt.axvline(x=(self.focal_plane - self.depth_of_field/2) * 1e6, color='g', linestyle=':')
        
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
        depth_of_field=2e-6,           # 2 micrometer depth of field
        diffusion_time=0.1,            # Time between frames
        num_particles=100,             # Number of particles
        gaussian_sigma=2.0,            # Base sigma for Gaussian
        brightness_factor=1.0,         # Brightness scaling factor
        snr_base=5.0,                  # Base SNR for smallest particle
        snr_scaling=2.0,               # SNR scaling with size
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