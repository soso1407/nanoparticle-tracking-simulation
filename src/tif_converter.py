"""
Nanoparticle GIF to TIF Converter

This module provides utility functions to convert GIF animations to TIF format
for the nanoparticle tracking simulation project.
"""

import os
from pathlib import Path
from PIL import Image
import glob
import numpy as np


def convert_gif_to_tif(gif_path, save_individual_frames=False):
    """
    Convert a GIF file to a 16-bit TIF file.
    
    Args:
        gif_path: Path to the GIF file
        save_individual_frames: If True, saves each frame as a separate TIF file
                               If False, saves as a multi-page TIF
                               
    Returns:
        Path to the created TIF file
    """
    # Ensure the file exists
    if not os.path.exists(gif_path):
        print(f"Error: File {gif_path} not found.")
        return None
    
    # Get the base name without extension
    base_path = os.path.splitext(gif_path)[0]
    tif_path = f"{base_path}.tif"
    
    try:
        # Open the GIF file
        with Image.open(gif_path) as img:
            # Check if it's animated
            frames = []
            
            # Extract all frames from the GIF
            try:
                while True:
                    # Convert to 16-bit grayscale
                    # First get to RGB if needed
                    if img.mode == 'RGBA':
                        frame = img.convert('RGB')
                    else:
                        frame = img.copy()
                    
                    # Convert to grayscale if not already
                    if frame.mode != 'L':
                        gray_frame = frame.convert('L')
                    else:
                        gray_frame = frame
                    
                    # Convert to 16-bit ('I;16')
                    # Scale values from 0-255 to 0-65535 for 16-bit range
                    array = np.array(gray_frame, dtype=np.uint16) * 257
                    frame_16bit = Image.fromarray(array, mode='I;16')
                    
                    frames.append(frame_16bit)
                    img.seek(img.tell() + 1)
            except EOFError:
                pass  # We've reached the end of the frames
            
            # Save frames based on the mode selected
            if save_individual_frames and len(frames) > 1:
                # Save each frame as a separate TIF file
                for i, frame in enumerate(frames):
                    frame_path = f"{base_path}_frame_{i:03d}.tif"
                    frame.save(frame_path, format='TIFF', compression='tiff_deflate')
                print(f"Saved {len(frames)} individual 16-bit TIF frames from {gif_path}")
                return base_path + "_frame_*.tif"
            else:
                # Save as a multi-page TIFF
                if frames:
                    frames[0].save(
                        tif_path, 
                        format='TIFF', 
                        compression='tiff_deflate',
                        save_all=True,
                        append_images=frames[1:] if len(frames) > 1 else []
                    )
                    print(f"Converted {gif_path} to 16-bit TIF: {tif_path}")
                    return tif_path
                else:
                    print(f"No frames found in {gif_path}")
                    return None
    except Exception as e:
        print(f"Error converting {gif_path} to TIF: {e}")
        return None


def save_frames_as_tif(frames, filename):
    """
    Save a list of frames as a 16-bit TIF file.
    
    Args:
        frames: List of PIL Images
        filename: Base filename to use (without extension)
    
    Returns:
        Path to the created TIF file
    """
    if not frames:
        print("No frames provided to save as TIF.")
        return None
    
    # Make sure the filename doesn't have an extension
    base_filename = os.path.splitext(filename)[0]
    tif_path = os.path.join('results', f"{base_filename}.tif")
    
    try:
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
        return tif_path
    except Exception as e:
        print(f"Error saving frames as TIF: {e}")
        return None


def convert_existing_gifs():
    """
    Convert all existing GIF files in the results directory to TIF format.
    """
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found.")
        return
    
    # Find all GIF files in the results directory
    gif_files = glob.glob(os.path.join(results_dir, "*.gif"))
    
    if not gif_files:
        print(f"No GIF files found in {results_dir}")
        return
    
    print(f"Found {len(gif_files)} GIF files to convert.")
    
    # Convert each GIF to TIF
    for gif_file in gif_files:
        convert_gif_to_tif(gif_file)
    
    print("Conversion complete.")


if __name__ == "__main__":
    # When run directly, convert all existing GIFs to TIFs
    convert_existing_gifs() 