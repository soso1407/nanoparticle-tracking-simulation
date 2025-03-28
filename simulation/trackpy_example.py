"""
TrackPy Integration Example

This script demonstrates how to use our track data with the TrackPy library.
It loads the simulation tracks and performs various TrackPy analyses.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import trackpy as tp
from track_utils import TrackConverter, calculate_track_statistics

def load_track_data(track_path):
    """
    Load track data from either JSON or CSV format.
    
    Args:
        track_path: Path to track file (either JSON or CSV)
    
    Returns:
        DataFrame containing track data
    """
    if track_path.endswith('.json'):
        # Load via our converter
        return TrackConverter.json_to_trackpy(track_path)
    elif track_path.endswith('.csv'):
        # Load directly
        return pd.read_csv(track_path)
    else:
        raise ValueError("Track file must be either .json or .csv")

def basic_trajectory_analysis(tracks_df, output_dir):
    """
    Perform basic trajectory analysis using TrackPy.
    
    Args:
        tracks_df: DataFrame containing track data
        output_dir: Directory to save output
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot trajectories
    plt.figure(figsize=(10, 10))
    tp.plot_traj(tracks_df, colorby='particle', cmap=plt.cm.jet)
    plt.title('Particle Trajectories')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.savefig(os.path.join(output_dir, 'trackpy_trajectories.png'), dpi=300)
    plt.close()
    
    # Compute Mean Squared Displacement
    im = tp.imsd(tracks_df, mpp=0.1, fps=10)  # 100nm per pixel, 10 fps
    
    # Plot MSD
    plt.figure(figsize=(10, 6))
    im.plot(style='.-', alpha=0.7)
    plt.xlabel('Lag time (s)')
    plt.ylabel('Mean squared displacement (μm$^2$)')
    plt.title('Mean Squared Displacement of Individual Particles')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'trackpy_msd.png'), dpi=300)
    plt.close()
    
    # Compute ensemble Mean Squared Displacement
    em = tp.emsd(tracks_df, mpp=0.1, fps=10)
    
    # Plot ensemble MSD
    plt.figure(figsize=(10, 6))
    em.plot(style='.-')
    plt.plot(em.index, em.index * 4 * 1e-3, 'k--', label='Slope = 1 (diffusive)')  # Theoretical line
    plt.xlabel('Lag time (s)')
    plt.ylabel('Ensemble mean squared displacement (μm$^2$)')
    plt.title('Ensemble Mean Squared Displacement')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'trackpy_ensemble_msd.png'), dpi=300)
    plt.close()
    
    return em

def size_based_analysis(tracks_df, output_dir):
    """
    Perform size-based analysis on the tracks.
    
    Args:
        tracks_df: DataFrame containing track data
        output_dir: Directory to save output
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by size (binning into small, medium, large)
    if 'size' in tracks_df.columns:
        # Get unique particles and their sizes (use the first instance of each particle)
        particle_sizes = tracks_df.groupby('particle')['size'].first()
        
        # Create size categories
        small = particle_sizes[particle_sizes < 90].index
        large = particle_sizes[particle_sizes > 110].index
        medium = particle_sizes[(particle_sizes >= 90) & (particle_sizes <= 110)].index
        
        # Create masks for each category
        small_mask = tracks_df['particle'].isin(small)
        medium_mask = tracks_df['particle'].isin(medium) 
        large_mask = tracks_df['particle'].isin(large)
        
        # Plot trajectories by size
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        tp.plot_traj(tracks_df[small_mask], colorby='particle', cmap=plt.cm.Reds)
        plt.title('Small Particles (<90 nm)')
        plt.xlabel('x (pixels)')
        plt.ylabel('y (pixels)')
        
        plt.subplot(1, 3, 2)
        tp.plot_traj(tracks_df[medium_mask], colorby='particle', cmap=plt.cm.Greens)
        plt.title('Medium Particles (90-110 nm)')
        plt.xlabel('x (pixels)')
        plt.ylabel('y (pixels)')
        
        plt.subplot(1, 3, 3)
        tp.plot_traj(tracks_df[large_mask], colorby='particle', cmap=plt.cm.Blues)
        plt.title('Large Particles (>110 nm)')
        plt.xlabel('x (pixels)')
        plt.ylabel('y (pixels)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trackpy_size_trajectories.png'), dpi=300)
        plt.close()
        
        # Calculate MSD for each size category
        em_small = tp.emsd(tracks_df[small_mask], mpp=0.1, fps=10) if any(small_mask) else None
        em_medium = tp.emsd(tracks_df[medium_mask], mpp=0.1, fps=10) if any(medium_mask) else None
        em_large = tp.emsd(tracks_df[large_mask], mpp=0.1, fps=10) if any(large_mask) else None
        
        # Plot MSD by size
        plt.figure(figsize=(10, 6))
        
        if em_small is not None:
            em_small.plot(style='r.-', label='Small (<90 nm)')
        if em_medium is not None:
            em_medium.plot(style='g.-', label='Medium (90-110 nm)')
        if em_large is not None:
            em_large.plot(style='b.-', label='Large (>110 nm)')
            
        plt.plot(em_small.index, em_small.index * 4 * 1e-3, 'k--', label='Slope = 1 (diffusive)')
        
        plt.xlabel('Lag time (s)')
        plt.ylabel('Ensemble mean squared displacement (μm$^2$)')
        plt.title('MSD by Particle Size')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'trackpy_size_msd.png'), dpi=300)
        plt.close()
        
        return {
            'small': em_small,
            'medium': em_medium,
            'large': em_large
        }
    else:
        print("Track data does not contain size information")
        return None

def brightness_analysis(tracks_df, output_dir):
    """
    Analyze the relationship between particle size and brightness.
    
    Args:
        tracks_df: DataFrame containing track data
        output_dir: Directory to save output
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    if 'size' in tracks_df.columns and 'mass' in tracks_df.columns:
        # Group by particle to get average size and brightness
        particle_stats = tracks_df.groupby('particle').agg({
            'size': 'mean',
            'mass': 'mean'
        }).reset_index()
        
        # Plot size vs brightness
        plt.figure(figsize=(10, 6))
        plt.scatter(particle_stats['size'], particle_stats['mass'], alpha=0.7)
        plt.xlabel('Particle Size (nm)')
        plt.ylabel('Brightness (mass)')
        plt.title('Particle Size vs Brightness')
        plt.grid(True, alpha=0.3)
        
        # Add best fit line
        if len(particle_stats) > 1:
            z = np.polyfit(particle_stats['size'], particle_stats['mass'], 1)
            p = np.poly1d(z)
            plt.plot(particle_stats['size'], p(particle_stats['size']), 'r--', 
                    label=f'Best Fit: y={z[0]:.2e}x+{z[1]:.2e}')
            plt.legend()
        
        plt.savefig(os.path.join(output_dir, 'trackpy_size_vs_brightness.png'), dpi=300)
        plt.close()
        
        return particle_stats
    else:
        print("Track data does not contain both size and brightness information")
        return None

def main():
    """Run TrackPy analysis on the simulation track data."""
    # Define paths
    track_csv = os.path.join('results', 'iteration_2', 'simulation_v2_tracks.csv')
    output_dir = os.path.join('results', 'iteration_2', 'trackpy_analysis')
    
    # Ensure track file exists
    if not os.path.exists(track_csv):
        print(f"Track file {track_csv} not found. Run simulation_v2.py first.")
        return
    
    # Load track data
    print(f"Loading track data from {track_csv}...")
    tracks_df = load_track_data(track_csv)
    print(f"Loaded {len(tracks_df)} track points for {tracks_df['particle'].nunique()} particles")
    
    # Perform basic analysis
    print("Performing basic trajectory analysis...")
    em = basic_trajectory_analysis(tracks_df, output_dir)
    
    # Perform size-based analysis
    print("Performing size-based analysis...")
    size_results = size_based_analysis(tracks_df, output_dir)
    
    # Perform brightness analysis
    print("Analyzing size vs brightness relationship...")
    brightness_results = brightness_analysis(tracks_df, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 