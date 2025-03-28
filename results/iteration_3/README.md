# Nanoparticle Tracking Simulation v3 - 3D with Focal Plane

## Overview

This iteration of the simulation introduces significant enhancements to create a more realistic representation of nanoparticle tracking in optical microscopy:

1. **3D Particle Tracking**: Particles now move in three dimensions (x, y, z) following 3D Brownian motion physics.
2. **Focal Plane Simulation**: A configurable focal plane has been implemented, where particles appear brightest when in focus and fade as they move away.
3. **Depth of Field Effects**: The simulation includes realistic depth of field effects, where particles become both dimmer and more blurred as they move out of the focal plane.
4. **Size-Dependent Rendering**: Larger particles have wider Gaussian profiles and appear brighter, accurately modeling their optical characteristics.
5. **Enhanced Optical Physics**: The simulation incorporates more realistic optical physics, including defocus blur and z-dependent brightness attenuation.

## Physical Model

The physical model has been extended to include:

- **3D Brownian Motion**: The Stokes-Einstein diffusion equation now operates in three dimensions with a 3D displacement vector.
- **Optical Depth Effects**: Particles' brightness attenuates according to a Gaussian function based on distance from the focal plane.
- **Defocus Blur**: Particles outside the focal plane appear more blurred, simulating the optical effects of defocus.
- **Volume-Based Brightness**: Particle brightness continues to scale with volume (r³), but now includes focal attenuation.

## Parameters

The simulation uses the following default parameters:

- **Temperature**: 298.15 K (room temperature)
- **Viscosity**: 1.0e-3 Pa·s (water)
- **Mean Particle Radius**: 50 nm
- **Standard Deviation of Particle Radius**: 10 nm
- **Frame Size**: 512 × 512 pixels
- **Pixel Size**: 100 nm
- **Z Range**: -10 to 10 micrometers
- **Focal Plane**: 0 micrometers (center of z-range)
- **Depth of Field**: 2 micrometers
- **Diffusion Time**: 0.1 seconds per frame
- **Number of Particles**: 100
- **Background Noise Level**: 0.02 (on a 0-1 scale)

## Output Files

The simulation generates the following output files:

- **simulation_v3.gif**: Animated GIF showing the 2D projection of the 3D simulation.
- **simulation_v3.tif**: 16-bit TIF file containing all frames with improved dynamic range.
- **simulation_v3_tracks.csv**: CSV file containing detailed track data for all particles, including 3D positions.
- **simulation_v3_tracks.json**: JSON file containing the same track data in structured format.
- **simulation_v3_metadata.json**: JSON file containing simulation parameters and metadata.
- **3d_positions.png**: 3D scatter plot showing particle positions with the focal plane.
- **depth_vs_brightness.png**: Plot showing the relationship between z-distance and brightness attenuation.
- **size_distribution.png**: Histogram showing the distribution of particle sizes.

## Track Data Fields

The track data now includes additional fields related to 3D positioning and focal plane effects:

- **frame**: Frame number
- **particle**: Particle ID
- **x, y, z**: 3D coordinates in pixels (x,y) and meters (z)
- **size**: Particle radius in meters
- **mass**: Particle volume (proportional to r³)
- **brightness**: Normalized brightness value (0-1)
- **raw_brightness**: Physical brightness value before normalization
- **brightness_uncertainty**: Uncertainty in brightness measurement
- **snr**: Signal-to-noise ratio
- **focal_attenuation**: Brightness attenuation due to distance from focal plane
- **diffusion_coefficient**: Particle's diffusion coefficient in m²/s

## 3-Particle Test Version

A special test version with only 3 particles is also available in the `results/test_3_particles_3d` directory. This version includes additional visualizations to help understand 3D movement and focal plane effects:

- **particle_3d_paths.png**: 3D plot showing the complete path of each particle.
- **z_positions_over_time.png**: Plot showing how particles move in the z-direction over time.
- **brightness_over_time.png**: Plot showing how brightness changes as particles move through the focal plane.

## How to Run

To run the 3D simulation:

```python
python src/simulation_v3.py
```

To run the 3-particle test version:

```python
python src/simulation_v3_test_3p.py
```

## Applications

This enhanced 3D simulation can be used to:

1. Test tracking algorithms' ability to handle particles moving in and out of focus.
2. Evaluate how depth of field affects particle detection and measurement.
3. Develop more robust tracking methods that account for 3D motion.
4. Validate experimental setups by comparing simulated and real microscopy data.
5. Train machine learning models on realistic 3D particle behavior. 