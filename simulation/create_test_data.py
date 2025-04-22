"""
Generate synthetic test data for simulation_v4.
Creates a TIF file with artificial particle data.
"""

import numpy as np
import tifffile
import os

def create_test_data(
    frame_size=(64, 64),  # Smaller frame size for faster testing
    num_particles=5,      # Fewer particles
    num_frames=3         # Fewer frames
):
    """Create synthetic microscopy data."""
    # Create output directory if it doesn't exist
    os.makedirs('test_data', exist_ok=True)
    
    # Initialize empty stack of frames
    frames = np.zeros((num_frames, *frame_size))  # Keep as float for now
    
    # Generate random particle positions
    x_positions = np.random.uniform(0, frame_size[0], num_particles)
    y_positions = np.random.uniform(0, frame_size[1], num_particles)
    
    # Generate frames
    for frame_idx in range(num_frames):
        # Create empty frame
        frame = np.zeros(frame_size)
        
        # Add Gaussian spots for each particle
        for i in range(num_particles):
            # Add small random movement
            x = x_positions[i] + np.random.normal(0, 1)
            y = y_positions[i] + np.random.normal(0, 1)
            
            # Create meshgrid for Gaussian
            y_grid, x_grid = np.meshgrid(
                np.arange(frame_size[1]),
                np.arange(frame_size[0])
            )
            
            # Add Gaussian spot
            sigma = 2.0
            intensity = 0.8
            gaussian = intensity * np.exp(
                -((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2)
            )
            
            frame += gaussian
        
        # Add some noise
        frame += np.random.normal(0, 0.02, frame_size)
        
        # Normalize to 0-1 range
        frame = (frame - frame.min()) / (frame.max() - frame.min())
        frames[frame_idx] = frame
    
    # Convert to 16-bit
    frames_16bit = (frames * 65535).astype(np.uint16)
    
    # Save as TIF
    output_path = 'test_data/synthetic_microscopy.tif'
    tifffile.imwrite(output_path, frames_16bit)
    print(f"Created test data at {output_path}")
    print(f"Shape: {frames_16bit.shape}")
    print(f"Value range: {frames_16bit.min()}-{frames_16bit.max()}")
    
    # Also save normalized version for MCMC fitting
    output_path_norm = 'test_data/synthetic_microscopy_normalized.npy'
    np.save(output_path_norm, frames)
    print(f"Saved normalized data at {output_path_norm}")
    
    return output_path

if __name__ == '__main__':
    create_test_data() 