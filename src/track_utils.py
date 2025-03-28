"""
Track Manipulation Utilities

This module provides utilities for manipulating and converting particle tracks
between different formats, with particular focus on TrackPy compatibility.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class TrackConverter:
    """
    Converts between different track data formats.
    
    Supports conversion between:
    - Our custom JSON format 
    - TrackPy DataFrame format
    - CSV files
    """
    
    @staticmethod
    def trackpy_to_json(df: pd.DataFrame, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Convert a TrackPy DataFrame to our JSON track format.
        
        Args:
            df: TrackPy DataFrame
            output_path: Optional path to save JSON output
        
        Returns:
            List of track dictionaries
        """
        if 'particle' not in df.columns:
            raise ValueError("DataFrame must contain a 'particle' column")
        
        # Get all unique particle IDs
        particle_ids = df['particle'].unique()
        
        # Create tracks for each particle
        tracks = []
        for pid in particle_ids:
            # Get data for this particle
            particle_data = df[df['particle'] == pid]
            
            # Create track dictionary
            track = {
                "particle_id": int(pid),
                "frames": particle_data['frame'].tolist(),
                "positions": list(zip(particle_data['x'].tolist(), particle_data['y'].tolist())),
            }
            
            # Add size if available
            if 'size' in particle_data.columns:
                track["sizes"] = particle_data['size'].tolist()
            
            # Add brightness/mass if available
            if 'mass' in particle_data.columns:
                track["brightnesses"] = particle_data['mass'].tolist()
            
            # Add any other columns as additional data
            for col in particle_data.columns:
                if col not in ['frame', 'particle', 'x', 'y', 'size', 'mass']:
                    track[col] = particle_data[col].tolist()
            
            tracks.append(track)
        
        # Save to file if requested
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(tracks, f, indent=2)
            
            print(f"Saved {len(tracks)} tracks to {output_path}")
        
        return tracks
    
    @staticmethod
    def json_to_trackpy(json_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert our JSON track format to a TrackPy-compatible DataFrame.
        
        Args:
            json_data: List of track dictionaries or path to JSON file
        
        Returns:
            TrackPy-compatible DataFrame
        """
        # Load from file if a string is provided
        if isinstance(json_data, str):
            with open(json_data, 'r') as f:
                json_data = json.load(f)
        
        # Process each track into a list of dictionaries
        rows = []
        for track in json_data:
            particle_id = track["particle_id"]
            
            # Process each frame
            for i, frame in enumerate(track["frames"]):
                row = {
                    "frame": frame,
                    "particle": particle_id,
                    "x": track["positions"][i][0],
                    "y": track["positions"][i][1],
                }
                
                # Add size if available
                if "sizes" in track and i < len(track["sizes"]):
                    row["size"] = track["sizes"][i]
                
                # Add brightness if available 
                if "brightnesses" in track and i < len(track["brightnesses"]):
                    row["mass"] = track["brightnesses"][i]
                
                # Add any additional fields
                for key, values in track.items():
                    if key not in ["particle_id", "frames", "positions", "sizes", "brightnesses"]:
                        if isinstance(values, list) and i < len(values):
                            row[key] = values[i]
                
                rows.append(row)
        
        # Create DataFrame
        return pd.DataFrame(rows)
    
    @staticmethod
    def csv_to_json(csv_path: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Convert a TrackPy CSV export to our JSON track format.
        
        Args:
            csv_path: Path to CSV file
            output_path: Optional path to save JSON output
        
        Returns:
            List of track dictionaries
        """
        # Load CSV as DataFrame
        df = pd.read_csv(csv_path)
        
        # Convert to JSON
        return TrackConverter.trackpy_to_json(df, output_path)
    
    @staticmethod
    def json_to_csv(json_data: List[Dict[str, Any]], output_path: str) -> pd.DataFrame:
        """
        Convert our JSON track format to a CSV file.
        
        Args:
            json_data: List of track dictionaries or path to JSON file
            output_path: Path to save CSV output
        
        Returns:
            TrackPy-compatible DataFrame
        """
        # Convert to DataFrame
        df = TrackConverter.json_to_trackpy(json_data)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Saved track data to {output_path}")
        
        return df

def calculate_track_statistics(tracks_json: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics from track data.
    
    Args:
        tracks_json: List of track dictionaries or path to JSON file
    
    Returns:
        Dictionary of statistics
    """
    # Load from file if a string is provided
    if isinstance(tracks_json, str):
        with open(tracks_json, 'r') as f:
            tracks_json = json.load(f)
    
    # Calculate statistics
    num_tracks = len(tracks_json)
    total_points = sum(len(track["frames"]) for track in tracks_json)
    
    # Calculate average track length
    avg_track_length = total_points / num_tracks if num_tracks > 0 else 0
    
    # Calculate size statistics if available
    sizes = []
    for track in tracks_json:
        if "sizes" in track:
            sizes.extend(track["sizes"])
    
    size_stats = {}
    if sizes:
        size_stats["mean"] = np.mean(sizes)
        size_stats["std"] = np.std(sizes)
        size_stats["min"] = np.min(sizes)
        size_stats["max"] = np.max(sizes)
    
    # Calculate brightness statistics if available
    brightnesses = []
    for track in tracks_json:
        if "brightnesses" in track:
            brightnesses.extend(track["brightnesses"])
    
    brightness_stats = {}
    if brightnesses:
        brightness_stats["mean"] = float(np.mean(brightnesses))
        brightness_stats["std"] = float(np.std(brightnesses))
        brightness_stats["min"] = float(np.min(brightnesses))
        brightness_stats["max"] = float(np.max(brightnesses))
    
    return {
        "num_tracks": num_tracks,
        "total_points": total_points,
        "avg_track_length": avg_track_length,
        "size_statistics": size_stats,
        "brightness_statistics": brightness_stats,
    }

def compare_tracks(ground_truth_json: str, measured_json: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Compare ground truth tracks with measured tracks.
    
    Args:
        ground_truth_json: Path to ground truth JSON file
        measured_json: Path to measured JSON file
        output_path: Optional path to save comparison results
    
    Returns:
        Dictionary of comparison metrics
    """
    # Load track data
    with open(ground_truth_json, 'r') as f:
        gt_tracks = json.load(f)
    
    with open(measured_json, 'r') as f:
        measured_tracks = json.load(f)
    
    # Convert to DataFrames for easier comparison
    gt_df = TrackConverter.json_to_trackpy(gt_tracks)
    measured_df = TrackConverter.json_to_trackpy(measured_tracks)
    
    # Basic statistics
    gt_stats = calculate_track_statistics(gt_tracks)
    measured_stats = calculate_track_statistics(measured_tracks)
    
    # Frame statistics
    gt_frames = set(gt_df['frame'].unique())
    measured_frames = set(measured_df['frame'].unique())
    
    # Compare number of particles per frame
    gt_particles_per_frame = gt_df.groupby('frame').size()
    measured_particles_per_frame = measured_df.groupby('frame').size()
    
    # For common frames, calculate differences
    common_frames = gt_frames.intersection(measured_frames)
    particles_diff = {}
    
    for frame in common_frames:
        gt_count = gt_particles_per_frame.get(frame, 0)
        measured_count = measured_particles_per_frame.get(frame, 0)
        particles_diff[int(frame)] = measured_count - gt_count
    
    # Prepare comparison results
    comparison = {
        "ground_truth": gt_stats,
        "measured": measured_stats,
        "frame_coverage": {
            "ground_truth_frames": len(gt_frames),
            "measured_frames": len(measured_frames),
            "common_frames": len(common_frames),
            "missing_frames": len(gt_frames - measured_frames),
            "extra_frames": len(measured_frames - gt_frames),
        },
        "particle_counts": {
            "average_diff_per_frame": np.mean(list(particles_diff.values())) if particles_diff else 0,
            "max_diff": max(particles_diff.values()) if particles_diff else 0,
            "min_diff": min(particles_diff.values()) if particles_diff else 0,
            "frame_differences": particles_diff
        }
    }
    
    # Save to file if requested
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"Saved comparison results to {output_path}")
    
    return comparison

# Example usage
if __name__ == "__main__":
    # This script can be run to test track conversion functionality
    print("Track Utilities Module")
    print("This module provides utilities for manipulating particle tracks.")
    print("Import this module to use its functions in your code.") 