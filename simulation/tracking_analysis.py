"""
Nanoparticle Tracking Analysis

This script analyzes the trajectories of simulated nanoparticles at different temperatures
and generates visualizations of the differences in particle movement.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
import matplotlib.patheffects as path_effects
from simulation import NanoparticleSimulator
from tif_converter import convert_gif_to_tif

class NanoparticleTracker:
    """
    Tracks and analyzes nanoparticle movement across different temperature simulations.
    """
    
    def __init__(self, frame_size=(2048, 2048), output_dir="results"):
        """
        Initialize the tracker.
        
        Args:
            frame_size: Size of simulation frame in pixels
            output_dir: Directory to save results
        """
        self.frame_size = frame_size
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create simulators for different temperatures
        self.simulators = {
            "cold": NanoparticleSimulator(
                num_particles=50,  # Fewer particles for cleaner visualization
                temperature=278.15,  # Cold temperature in Kelvin (5°C)
                viscosity=1.5e-3,    # Higher viscosity at lower temperature
                particle_radius=50e-9,
                frame_size=frame_size,
                pixel_size=100e-9,
                diffusion_time=0.01,
                gaussian_sigma=2.0
            ),
            "room": NanoparticleSimulator(
                num_particles=50,
                temperature=298.15,  # Room temperature in Kelvin (25°C)
                viscosity=1.0e-3,
                particle_radius=50e-9,
                frame_size=frame_size,
                pixel_size=100e-9,
                diffusion_time=0.01,
                gaussian_sigma=2.0
            ),
            "hot": NanoparticleSimulator(
                num_particles=50,
                temperature=318.15,  # Hot temperature in Kelvin (45°C)
                viscosity=0.6e-3,    # Lower viscosity at higher temperature
                particle_radius=50e-9,
                frame_size=frame_size,
                pixel_size=100e-9,
                diffusion_time=0.01,
                gaussian_sigma=2.0
            )
        }
        
        # Dictionary to store trajectory data
        self.trajectories = {}
        
    def track_particles(self, num_frames=100):
        """
        Generate and track particle positions for a specified number of frames.
        
        Args:
            num_frames: Number of frames to simulate
        """
        # Track particles for each temperature condition
        for temp_name, simulator in self.simulators.items():
            print(f"Tracking particles for {temp_name} temperature simulation...")
            
            # Reset particle positions
            simulator.positions = np.random.rand(simulator.num_particles, 2) * simulator.frame_size
            
            # Initialize trajectory storage
            particle_trajectories = [[] for _ in range(simulator.num_particles)]
            
            # Track particles over time
            for _ in range(num_frames):
                # Update positions
                positions = simulator.step()
                
                # Store positions for each particle
                for i, pos in enumerate(positions):
                    particle_trajectories[i].append(pos.copy())
            
            # Convert to numpy arrays for easier handling
            self.trajectories[temp_name] = [np.array(traj) for traj in particle_trajectories]
            
            print(f"  Completed tracking {simulator.num_particles} particles over {num_frames} frames")
    
    def analyze_displacement(self):
        """
        Analyze and visualize the displacement of particles across temperature conditions.
        """
        if not self.trajectories:
            print("No trajectory data available. Run track_particles() first.")
            return
        
        # Calculate mean squared displacement for each temperature
        temp_colors = {"cold": "blue", "room": "green", "hot": "red"}
        temp_labels = {"cold": "Cold (5°C)", "room": "Room (25°C)", "hot": "Hot (45°C)"}
        
        plt.figure(figsize=(12, 8))
        
        for temp_name, trajectories in self.trajectories.items():
            # Calculate displacement from starting position for each particle
            displacements = []
            
            for trajectory in trajectories:
                # Get initial position
                initial_pos = trajectory[0]
                
                # Calculate displacement at each time point
                traj_displacement = np.sqrt(np.sum((trajectory - initial_pos)**2, axis=1))
                displacements.append(traj_displacement)
            
            # Calculate mean and standard deviation of displacement
            mean_displacement = np.mean(displacements, axis=0)
            std_displacement = np.std(displacements, axis=0)
            
            # Time points (in simulation time units)
            time_points = np.arange(len(mean_displacement)) * self.simulators[temp_name].diffusion_time
            
            # Plot mean displacement with error region
            plt.plot(time_points, mean_displacement, color=temp_colors[temp_name], 
                     label=temp_labels[temp_name], linewidth=2)
            plt.fill_between(time_points, 
                            mean_displacement - std_displacement, 
                            mean_displacement + std_displacement, 
                            color=temp_colors[temp_name], alpha=0.3)
        
        plt.xlabel("Time (s)")
        plt.ylabel("Mean Displacement (pixels)")
        plt.title("Nanoparticle Displacement vs. Time at Different Temperatures")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(self.output_dir, "displacement_comparison.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Displacement analysis saved to {output_path}")
    
    def visualize_trajectories(self):
        """
        Create static visualization of particle trajectories.
        """
        if not self.trajectories:
            print("No trajectory data available. Run track_particles() first.")
            return
        
        # Create a figure with subplots for each temperature
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        temp_names = ["cold", "room", "hot"]
        temp_labels = ["Cold (5°C)", "Room (25°C)", "Hot (45°C)"]
        
        for i, (temp_name, ax) in enumerate(zip(temp_names, axes)):
            ax.set_title(temp_labels[i])
            
            # Plot boundaries
            ax.set_xlim(0, self.frame_size[0])
            ax.set_ylim(0, self.frame_size[1])
            
            # Invert y-axis to match image coordinates
            ax.invert_yaxis()
            
            # Plot trajectories for this temperature
            for j, trajectory in enumerate(self.trajectories[temp_name]):
                # Only plot a subset of trajectories for clarity
                if j % 5 == 0:  # Plot every 5th particle
                    # Color gradient based on time
                    for k in range(len(trajectory) - 1):
                        progress = k / (len(trajectory) - 1)
                        color = plt.cm.viridis(progress)
                        ax.plot(trajectory[k:k+2, 0], trajectory[k:k+2, 1], 
                                color=color, linewidth=1, alpha=0.7)
                    
                    # Mark starting positions
                    ax.plot(trajectory[0, 0], trajectory[0, 1], 'o', 
                            color='blue', markersize=3)
                    
                    # Mark ending positions
                    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 's', 
                            color='red', markersize=3)
            
            # Add a time color bar
            if i == 2:  # Add to the last subplot only
                sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=axes.ravel().tolist())
                cbar.set_label('Time Progression')
        
        # Layout adjustments
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(self.output_dir, "trajectory_comparison.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Trajectory visualization saved to {output_path}")
    
    def create_animated_visualization(self, num_frames=100):
        """
        Create an animated visualization of particle movement at different temperatures.
        
        Args:
            num_frames: Number of frames for the animation
        """
        if not self.trajectories:
            print("No trajectory data available. Run track_particles() first.")
            return
        
        # Create a figure with subplots for each temperature
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        temp_names = ["cold", "room", "hot"]
        temp_labels = ["Cold (5°C)", "Room (25°C)", "Hot (45°C)"]
        
        # Customize figure appearance
        fig.patch.set_facecolor('#f0f0f0')
        
        # Set up each subplot
        for i, (temp_name, ax) in enumerate(zip(temp_names, axes)):
            title = ax.set_title(temp_labels[i], fontsize=14, fontweight='bold')
            title.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
            
            # Set limits to frame size
            ax.set_xlim(0, self.frame_size[0])
            ax.set_ylim(0, self.frame_size[1])
            ax.invert_yaxis()  # Invert y-axis to match image coordinates
            
            # Remove axes ticks for cleaner visualization
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
        
        # Text for showing frame number
        time_text = fig.text(0.02, 0.02, '', fontsize=12)
        
        # Set up scatter plots for each temperature
        scatters = []
        trails = []
        
        for i, temp_name in enumerate(temp_names):
            # Initial positions for scatter plot
            positions = np.array([traj[0] for traj in self.trajectories[temp_name]])
            
            # Create scatter plot for current positions
            scatter = axes[i].scatter(
                positions[:, 0], positions[:, 1], 
                c='white', edgecolor='black', s=30, alpha=0.8
            )
            scatters.append(scatter)
            
            # Create line objects for trails
            trail_lines = []
            for j in range(len(positions)):
                line, = axes[i].plot([], [], color='cyan', alpha=0.5, linewidth=1)
                trail_lines.append(line)
            trails.append(trail_lines)
        
        # Update function for animation
        def update(frame):
            # Update time text
            time = frame * self.simulators["room"].diffusion_time
            time_text.set_text(f'Simulation Time: {time:.2f} s')
            
            # Update each temperature simulation
            for i, temp_name in enumerate(temp_names):
                # Get positions for current frame
                positions = np.array([traj[frame] for traj in self.trajectories[temp_name]])
                
                # Update scatter positions
                scatters[i].set_offsets(positions)
                
                # Update trailing lines (show previous 10 positions)
                trail_length = 10
                start_idx = max(0, frame - trail_length)
                
                for j, line in enumerate(trails[i]):
                    if frame > 0:
                        trajectory = self.trajectories[temp_name][j]
                        line.set_data(
                            trajectory[start_idx:frame+1, 0],
                            trajectory[start_idx:frame+1, 1]
                        )
            
            return scatters + [item for sublist in trails for item in sublist] + [time_text]
        
        # Create animation
        ani = FuncAnimation(
            fig, update, frames=num_frames,
            interval=100, blit=True
        )
        
        # Save animation
        output_path = os.path.join(self.output_dir, "temperature_animation.gif")
        ani.save(output_path, writer='pillow', fps=10)
        
        print(f"Animated visualization saved to {output_path}")
        
        # Also save as TIF
        tif_path = convert_gif_to_tif(output_path)
        if tif_path:
            print(f"TIF version saved to {tif_path}")

def main():
    """Run nanoparticle tracking analysis."""
    # Initialize tracker
    tracker = NanoparticleTracker()
    
    # Track particles
    tracker.track_particles(num_frames=100)
    
    # Analyze and visualize results
    tracker.analyze_displacement()
    tracker.visualize_trajectories()
    tracker.create_animated_visualization()

if __name__ == "__main__":
    main() 