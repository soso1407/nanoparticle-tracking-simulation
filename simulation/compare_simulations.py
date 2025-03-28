"""
Nanoparticle Simulation Comparison

This script creates a side-by-side comparison of nanoparticle simulations at different temperatures.
It combines frames from three separate simulations to create a single comparison video.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_comparison_gif(output_path="results/temperature_comparison.gif"):
    """
    Create a side-by-side comparison GIF from three temperature simulations.
    
    Args:
        output_path: Path to save the output GIF
    """
    # Input paths
    cold_frames_path = "results/simulation_cold_temp.gif"
    room_frames_path = "results/simulation_room_temp.gif"
    hot_frames_path = "results/simulation_hot_temp.gif"
    
    # Check if input files exist
    for path in [cold_frames_path, room_frames_path, hot_frames_path]:
        if not os.path.exists(path):
            print(f"Error: {path} does not exist. Run simulation.py first.")
            return
    
    # Open the GIFs and extract frames
    try:
        cold_gif = Image.open(cold_frames_path)
        room_gif = Image.open(room_frames_path)
        hot_gif = Image.open(hot_frames_path)
        
        # Count frames (use the minimum frame count)
        n_frames = min(
            get_frame_count(cold_gif),
            get_frame_count(room_gif),
            get_frame_count(hot_gif)
        )
        
        # Prepare frames for combined GIF
        combined_frames = []
        
        # Try to load a font for labels
        try:
            font = ImageFont.truetype("Arial", 40)
        except IOError:
            # If Arial isn't available, use default font
            font = ImageFont.load_default()
        
        # Process each frame
        for i in range(n_frames):
            # Get frames from each simulation
            cold_frame = extract_frame(cold_gif, i)
            room_frame = extract_frame(room_gif, i)
            hot_frame = extract_frame(hot_gif, i)
            
            # Resize frames if needed (optional for consistency)
            frame_size = (512, 512)
            cold_frame = cold_frame.resize(frame_size)
            room_frame = room_frame.resize(frame_size)
            hot_frame = hot_frame.resize(frame_size)
            
            # Create combined frame
            width = frame_size[0] * 3 + 20  # 3 frames + padding
            height = frame_size[1] + 80  # Frame height + space for labels
            combined = Image.new('RGB', (width, height), (10, 10, 10))
            
            # Paste frames
            combined.paste(cold_frame, (10, 70))
            combined.paste(room_frame, (frame_size[0] + 10, 70))
            combined.paste(hot_frame, (frame_size[0] * 2 + 10, 70))
            
            # Add frame numbers and temperature labels
            draw = ImageDraw.Draw(combined)
            
            # Frame number
            draw.text((10, 10), f"Frame {i+1}/{n_frames}", fill=(255, 255, 255), font=font)
            
            # Temperature labels
            draw.text((frame_size[0]//2 - 50, 40), "Cold (5°C)", fill=(100, 100, 255), font=font)
            draw.text((frame_size[0] + frame_size[0]//2 - 60, 40), "Room (25°C)", fill=(200, 200, 200), font=font)
            draw.text((2 * frame_size[0] + frame_size[0]//2 - 50, 40), "Hot (45°C)", fill=(255, 100, 100), font=font)
            
            combined_frames.append(combined)
            
            # Reset GIFs for next frame
            cold_gif.seek(i % get_frame_count(cold_gif))
            room_gif.seek(i % get_frame_count(room_gif))
            hot_gif.seek(i % get_frame_count(hot_gif))
            
            # Show progress
            if i % 10 == 0:
                print(f"Processing frame {i}/{n_frames}")
        
        # Save combined GIF
        print(f"Saving comparison GIF to {output_path}")
        combined_frames[0].save(
            output_path,
            save_all=True,
            append_images=combined_frames[1:],
            duration=100,  # Slightly slower for better viewing
            loop=0,
            optimize=False
        )
        
        print("Comparison GIF created successfully!")
        
    except Exception as e:
        print(f"Error creating comparison GIF: {e}")
    finally:
        # Close GIFs
        try:
            cold_gif.close()
            room_gif.close()
            hot_gif.close()
        except:
            pass

def get_frame_count(gif):
    """
    Count the number of frames in a GIF.
    
    Args:
        gif: PIL Image object of a GIF
        
    Returns:
        Number of frames
    """
    # Save current position
    current_pos = gif.tell()
    
    # Count frames
    frames = 0
    try:
        while True:
            frames += 1
            gif.seek(frames)
    except EOFError:
        pass
    
    # Return to original position
    gif.seek(current_pos)
    
    return frames

def extract_frame(gif, frame_index):
    """
    Extract a specific frame from a GIF.
    
    Args:
        gif: PIL Image object of a GIF
        frame_index: Index of the frame to extract
        
    Returns:
        PIL Image of the extracted frame
    """
    try:
        gif.seek(frame_index)
        return gif.copy().convert("RGB")
    except EOFError:
        # If frame index is out of bounds, return the first frame
        gif.seek(0)
        return gif.copy().convert("RGB")

def create_trajectory_comparison():
    """
    Create a comparison of particle trajectories from different temperature simulations.
    """
    # This would require tracking specific particles throughout the simulation
    # Not implemented in the current version, as it requires modifications to the 
    # simulation code to track and export particle trajectories
    pass

if __name__ == "__main__":
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Create comparison GIF
    create_comparison_gif() 