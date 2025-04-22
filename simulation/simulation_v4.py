"""
SIMULATION V4: MCMC-Enhanced Nanoparticle Tracking

New features in version 4:
- MCMC fitting of Gaussian blurs to real data
- PyMC integration for parameter estimation
- Gaussian error modeling in pixel brightnesses
- Enhanced observation equation with flexible distributions
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
from typing import List, Tuple, Optional, Dict, Any
import os
import json
import pandas as pd
import tifffile
from simulation_v3 import NanoparticleSimulator3D, ParticleTrack3D
import time
from pathlib import Path
import argparse

class MCMCFitter:
    """Class for fitting Gaussian parameters to real data using MCMC."""
    
    def __init__(self, 
                 real_data: np.ndarray,
                 num_particles: int = 100,
                 pixel_size: float = 100e-9,
                 frame_size: Tuple[int, int] = (512, 512)):
        """Initialize the MCMC fitter."""
        # Take first frame if multiple frames provided
        if len(real_data.shape) == 3:
            self.real_data = real_data[0]
        else:
            self.real_data = real_data
        
        # Normalize data to 0-1 range
        self.real_data = (self.real_data - self.real_data.min()) / (self.real_data.max() - self.real_data.min())
        
        self.num_particles = num_particles
        self.pixel_size = pixel_size
        self.frame_size = self.real_data.shape
        self.model = None
        self.trace = None
    
    def create_model(self, 
                    sigma_prior_mean: float = 2.0,
                    intensity_prior_mean: float = 0.8):
        """Create PyMC model for parameter estimation."""
        print("Creating MCMC model...")
        with pm.Model() as self.model:
            # Priors for particle parameters
            x_positions = pm.Uniform('x_positions', 
                                   lower=0, 
                                   upper=self.frame_size[0], 
                                   shape=self.num_particles)
            
            y_positions = pm.Uniform('y_positions', 
                                   lower=0, 
                                   upper=self.frame_size[1], 
                                   shape=self.num_particles)
            
            # Prior for Gaussian width (sigma)
            sigma = pm.TruncatedNormal('sigma',
                                     mu=sigma_prior_mean,
                                     sigma=0.5,
                                     lower=0.5,
                                     upper=5.0)
            
            # Prior for particle intensities
            intensities = pm.TruncatedNormal('intensities',
                                           mu=intensity_prior_mean,
                                           sigma=0.2,
                                           lower=0.0,
                                           upper=1.0,
                                           shape=self.num_particles)
            
            # Generate synthetic image
            synthetic_image = self._generate_synthetic_image(
                x_positions, y_positions, sigma, intensities)
            
            # Simple Gaussian likelihood with larger sigma for stability
            pm.Normal('obs',
                     mu=synthetic_image,
                     sigma=0.2,  # Increased observation noise
                     observed=self.real_data)
        print("Model created successfully")
    
    def _generate_synthetic_image(self,
                                x_positions,
                                y_positions,
                                sigma,
                                intensities) -> np.ndarray:
        """Generate synthetic image from particle parameters."""
        # Create coordinate grids
        x = np.arange(self.frame_size[0])
        y = np.arange(self.frame_size[1])
        X, Y = np.meshgrid(x, y)
        
        # Initialize image
        image = pm.math.zeros(self.frame_size)
        
        # Add particles one by one
        for i in range(self.num_particles):
            dx = X - x_positions[i]
            dy = Y - y_positions[i]
            r2 = dx**2 + dy**2
            gaussian = intensities[i] * pm.math.exp(-r2 / (2 * sigma**2))
            image = image + gaussian
        
        return image
    
    def fit(self, 
            num_samples: int = 50,  # Reduced from 200 to 50 for faster testing
            num_chains: int = 1,    # Reduced from 2 to 1
            tune: int = 25) -> None:  # Reduced from 100 to 25
        """Run MCMC sampling to fit parameters."""
        print(f"Starting MCMC fitting with {num_samples} samples and {num_chains} chains...")
        start_time = time.time()
        
        with self.model:
            # Use NUTS sampler with progress bar
            self.trace = pm.sample(
                draws=num_samples,
                chains=num_chains,
                tune=tune,
                return_inferencedata=True,
                progressbar=True
            )
        
        elapsed_time = time.time() - start_time
        print(f"MCMC fitting completed in {elapsed_time:.1f} seconds")
    
    def get_fitted_parameters(self) -> Dict[str, np.ndarray]:
        """Get mean values of fitted parameters."""
        if self.trace is None:
            raise ValueError("Must run fit() before getting parameters")
        
        print("Extracting fitted parameters...")
        params = {
            'x_positions': np.mean(self.trace.posterior['x_positions'], axis=(0, 1)),
            'y_positions': np.mean(self.trace.posterior['y_positions'], axis=(0, 1)),
            'sigma': float(np.mean(self.trace.posterior['sigma'])),
            'intensities': np.mean(self.trace.posterior['intensities'], axis=(0, 1))
        }
        print("Parameters extracted successfully")
        return params
    
    def plot_diagnostics(self) -> None:
        """Plot MCMC diagnostics using ArviZ."""
        if self.trace is None:
            raise ValueError("Must run fit() before plotting diagnostics")
        
        print("Generating diagnostic plots...")
        # Create output directory
        os.makedirs('results/mcmc_diagnostics', exist_ok=True)
        
        # Plot trace
        az.plot_trace(self.trace, var_names=['sigma'])
        plt.savefig('results/mcmc_diagnostics/trace_plot.png')
        plt.close()
        
        # Plot rank
        az.plot_rank(self.trace, var_names=['sigma'])
        plt.savefig('results/mcmc_diagnostics/rank_plot.png')
        plt.close()
        
        print("Diagnostic plots saved in results/mcmc_diagnostics/")

class FittedSimulator(NanoparticleSimulator3D):
    """Enhanced simulator with MCMC-fitted parameters."""
    
    def __init__(self,
                 fitted_parameters: Dict[str, np.ndarray],
                 brightness_error_std: float = 0.1,
                 *args, **kwargs):
        """
        Initialize the fitted simulator.
        
        Args:
            fitted_parameters: Parameters from MCMC fitting
            brightness_error_std: Standard deviation for Gaussian brightness error
            *args, **kwargs: Arguments for parent class
        """
        super().__init__(*args, **kwargs)
        
        self.fitted_parameters = fitted_parameters
        self.brightness_error_std = brightness_error_std
        
        # Override particle positions with fitted values
        self.positions = np.column_stack([
            fitted_parameters['x_positions'] * self.pixel_size,
            fitted_parameters['y_positions'] * self.pixel_size,
            np.zeros(self.num_particles)  # Start at focal plane
        ])
    
    def _apply_gaussian_error(self, frame: np.ndarray) -> np.ndarray:
        """Apply Gaussian error to frame intensities."""
        error = np.random.normal(0, self.brightness_error_std, frame.shape)
        return np.clip(frame + error, 0, 1)
    
    def run_simulation(self, num_frames: int = 100) -> Tuple[List[Image.Image], np.ndarray]:
        """
        Run simulation with fitted parameters.
        
        Args:
            num_frames: Number of frames to simulate
            
        Returns:
            Tuple of (8-bit frames for GIF, 16-bit frames for TIF)
        """
        frames, frames_16bit = super().run_simulation(num_frames)
        
        # Apply additional Gaussian error
        frames_16bit = np.array([
            self._apply_gaussian_error(frame) for frame in frames_16bit
        ])
        
        return frames, frames_16bit
    
    def save_fitted_parameters(self, filename: str) -> None:
        """Save fitted parameters to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        json_data = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in self.fitted_parameters.items()
        }
        
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)

def generate_synthetic_test_data(num_frames=10, frame_size=(64, 64)):
    """Generate synthetic microscopy data for testing.
    
    Returns:
        np.ndarray: Synthetic data with shape (num_frames, height, width)
    """
    print("Generating synthetic test data...")
    data = np.zeros((num_frames, *frame_size))
    
    # Add some particles with Gaussian profiles
    for _ in range(5):  # 5 particles
        x = np.random.randint(10, frame_size[0]-10)
        y = np.random.randint(10, frame_size[1]-10)
        sigma = np.random.uniform(1.5, 3.0)
        intensity = np.random.uniform(0.5, 1.0)
        
        x_grid, y_grid = np.meshgrid(np.arange(frame_size[0]), np.arange(frame_size[1]))
        gaussian = intensity * np.exp(-((x_grid - x)**2 + (y_grid - y)**2)/(2*sigma**2))
        
        for f in range(num_frames):
            # Add random walk motion
            x += np.random.normal(0, 1)
            y += np.random.normal(0, 1)
            x = np.clip(x, 10, frame_size[0]-10)
            y = np.clip(y, 10, frame_size[1]-10)
            
            frame_gaussian = intensity * np.exp(-((x_grid - x)**2 + (y_grid - y)**2)/(2*sigma**2))
            data[f] += frame_gaussian
    
    # Add noise
    data += np.random.normal(0, 0.05, size=data.shape)
    data = np.clip(data, 0, 1)
    
    return data

def fit_and_simulate(real_data_path, output_dir, num_particles=5, num_frames=10):
    """Fit parameters to real data and run simulation."""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or generate data
    if real_data_path is None:
        print("No input data provided. Generating synthetic test data...")
        real_data = generate_synthetic_test_data(num_frames=num_frames)
        np.save(output_dir / "synthetic_test_data.npy", real_data)
        print(f"Saved synthetic test data to {output_dir}/synthetic_test_data.npy")
    else:
        print(f"Loading data from {real_data_path}")
        real_data = np.load(real_data_path)
    
    print("Fitting parameters to real data...")
    # Fit parameters
    fitter = MCMCFitter(real_data, num_particles=num_particles)
    fitter.create_model()
    fitter.fit()
    
    # Get fitted parameters
    fitted_params = fitter.get_fitted_parameters()
    
    print("Running simulation with fitted parameters...")
    # Create and run fitted simulator
    simulator = FittedSimulator(
        fitted_parameters=fitted_params,
        num_particles=num_particles
    )
    
    frames, frames_16bit = simulator.run_simulation(num_frames)
    
    # Save simulation outputs
    print("Saving outputs...")
    tifffile.imwrite(
        os.path.join(output_dir, 'simulation_fitted.tif'),
        frames_16bit
    )
    
    frames[0].save(
        os.path.join(output_dir, 'simulation_fitted.gif'),
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )
    
    # Save fitted parameters
    params_path = os.path.join(output_dir, 'fitted_parameters.json')
    simulator.save_fitted_parameters(params_path)
    print(f"Saved fitted parameters to {params_path}")
    
    # Save MCMC diagnostics
    print("Generating diagnostic plots...")
    fitter.plot_diagnostics()
    
    print(f"Simulation complete. Results saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run nanoparticle tracking simulation with MCMC fitting")
    parser.add_argument("input_data", nargs="?", default=None, 
                      help="Path to real microscopy data (.npy file). If not provided, synthetic data will be generated.")
    parser.add_argument("--output-dir", default="results/mcmc_fitted_sim",
                      help="Directory to save results")
    parser.add_argument("--particles", type=int, default=5,
                      help="Number of particles to simulate")
    parser.add_argument("--frames", type=int, default=10,
                      help="Number of frames to simulate")
    
    args = parser.parse_args()
    
    fit_and_simulate(
        real_data_path=args.input_data,
        output_dir=args.output_dir,
        num_particles=args.particles,
        num_frames=args.frames
    ) 