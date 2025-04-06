# Nanoparticle Tracking Simulation

A comprehensive simulation tool for generating synthetic nanoparticle tracking data with realistic Brownian motion and optical physics. The simulation has evolved through three major iterations, each adding more sophisticated features to increase the realism and usefulness of the generated data.

## Project Overview

This project simulates nanoparticles moving in fluid via Brownian motion, as would be observed in an optical microscope. The simulation:

1. Models the physics of nanoparticle diffusion based on the Stokes-Einstein equation
2. Accounts for particle size variation and its effect on diffusion speed
3. Renders particles with realistic optical properties 
4. In the latest version, simulates 3D movement and focal plane effects

The simulation produces both visual outputs (GIF and TIF image sequences) and structured data files (CSV and JSON) for scientific analysis.

## Simulation Iterations

### Iteration 1: Basic 2D Simulation
- Simple 2D Brownian motion simulation
- Fixed particle sizes
- Circle-based particle rendering
- Temperature-dependent diffusion
- GIF output

### Iteration 2: Enhanced Brightness and Data Export
- Variable particle sizes following normal distribution
- Enhanced brightness modeling based on particle volume (r³)
- SNR and brightness uncertainty calculation
- Gaussian particle rendering
- 16-bit TIF output for higher dynamic range
- Track data export in CSV and JSON formats

### Iteration 3: Full 3D Simulation with Focal Plane
- Complete 3D particle tracking with z-position
- Focal plane simulation with configurable depth of field
- Z-dependent brightness attenuation
- Defocus blur for particles outside the focal plane
- Size-dependent rendering with realistic optical effects
- Enhanced visualization plots showing 3D trajectories

## Features

- **Advanced Brownian Motion Simulation**: Accurately models nanoparticle diffusion based on the Stokes-Einstein equation, considering particle size, temperature, and fluid viscosity.
- **Size Distribution Control**: Customizable normal distribution of particle sizes.
- **Realistic Brightness Modeling**:
  - Brightness scaling with volume (r³)
  - Direct 2D Gaussian rendering with peak intensities
  - Signal-to-noise ratio (SNR) estimates and brightness uncertainty modeling
  - Fluctuating brightness based on uncertainty
- **3D Simulation with Focal Plane** (v3):
  - Full 3D tracking of particles with realistic distribution
  - Configurable focal plane with realistic depth of field effects
  - Z-dependent brightness attenuation using multi-stage falloff
  - Defocus blur for out-of-focus particles
  - Stratified z-distribution for balanced particle visibility
- **Data Formats**:
  - Animated GIF for visualization
  - 16-bit TIF output for higher dynamic range analysis
  - CSV and JSON data export compatible with TrackPy and other analysis tools
- **Visualization Tools**:
  - 3D position plots
  - Particle path visualization
  - Depth vs. brightness plots
  - Size distribution histograms
  - Z-position vs. time plots
  - Brightness over time plots

## Physics Background

### Brownian Motion

The simulation is based on the Stokes-Einstein equation for diffusion:

```
D = (kB * T) / (6 * π * η * r)
```

Where:
- D is the diffusion coefficient (m²/s)
- kB is the Boltzmann constant (1.380649 × 10^-23 J/K)
- T is the temperature in Kelvin
- η is the fluid viscosity (Pa·s)
- r is the particle radius (m)

For 2D Brownian motion, the mean squared displacement (MSD) is given by:
```
MSD = 4 × D × t
```

For 3D Brownian motion, displacements in each dimension are calculated independently, with standard deviation:
```
σ = sqrt(2 * D * dt)
```

### Optical Imaging

The simulation models several key aspects of optical microscopy:

1. **Brightness Scaling**: Particle brightness scales with volume (r³), reflecting that larger particles scatter more light.
2. **Diffraction-Limited Imaging**: Particles are rendered as Gaussian spots with size reflecting both particle size and optical diffraction limits.
3. **Focal Plane Physics**: Particles appear brightest when in the focal plane and gradually fade as they move away in the z-direction.
4. **Depth of Field**: A configurable depth of field controls how quickly particles fade and blur as they move out of focus.
5. **Defocus Blur**: Particles outside the focal plane appear more blurred, with the blur increasing with distance.
6. **SNR Scaling**: Signal-to-noise ratio increases with particle size, reflecting stronger signals from larger particles.
7. **Measurement Uncertainty**: Brightness fluctuations model realistic measurement uncertainty based on SNR.

## Default Parameters

The simulation uses the following default parameters, which can be customized:

- **Temperature**: 298.15 K (room temperature)
- **Viscosity**: 1.0e-3 Pa·s (water viscosity)
- **Mean Particle Radius**: 50 nm
- **Standard Deviation of Particle Radius**: 10 nm (iteration 3), 4.5 nm (earlier iterations)
- **Frame Size**: 512 × 512 pixels (iteration 3), 2048 × 2048 pixels (iteration 2)
- **Pixel Size**: 100 nm
- **Z Range** (iteration 3): -10 to 10 micrometers
- **Focal Plane** (iteration 3): 0 micrometers (center of z-range)
- **Depth of Field** (iteration 3): 2 micrometers
- **Diffusion Time**: 0.1 seconds (iteration 3), 0.01 seconds (earlier iterations)
- **Number of Particles**: 100 (default), 3 (test simulations)
- **Background Noise Level**: 0.02 (on a 0-1 scale)

## Usage

Run the simulation:

```bash
# For the latest 3D simulation
python src/simulation_v3.py

# For the test version with 3 particles
python src/simulation_v3_test_3p.py

# For earlier versions
python src/simulation_v2.py
python src/simulation_v1.py
```

## Output Files

Each simulation run produces:

### Common Outputs
- Animated GIF visualization
- 16-bit TIF image stack
- CSV track data (compatible with TrackPy)
- JSON track data
- Parameter metadata (JSON)
- Size distribution plot (PNG)

### Iteration 3 Additional Outputs
- 3D positions plot
- Depth vs. brightness plot

### Test 3-Particle Version Additional Outputs
- 3D particle paths plot
- Z-positions over time plot
- Brightness over time plot

## Track Data Format

### JSON Format
The JSON track format uses the following structure:

```json
[
  {
    "particle_id": 0,
    "frames": [0, 1, 2, ...],
    "positions": [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2], ...],
    "sizes": [101.2, 101.2, 101.2, ...],
    "brightnesses": [0.8, 0.8, 0.8, ...],
    "raw_brightness": [7.85e-15, 7.85e-15, ...],
    "brightness_uncertainty": [0.08, 0.08, ...],
    "snr": [10.2, 10.2, ...],
    "diffusion_coefficient": [2.1e-12, 2.1e-12, ...],
    "focal_attenuation": [0.95, 0.87, 0.76, ...] // Iteration 3 only
  },
  {
    "particle_id": 1,
    ...
  }
]
```

### CSV Format (TrackPy Compatible)

The CSV format follows TrackPy's convention with additional fields:

| frame | particle | x | y | z | size | mass | brightness | raw_brightness | brightness_uncertainty | snr | diffusion_coefficient | focal_attenuation |
|-------|----------|---|---|---|------|------|------------|----------------|------------------------|-----|----------------------|-------------------|
| 0 | 0 | 423.1 | 1024.7 | 0.2e-6 | 98.5 | 0.75 | 0.95 | 7.85e-15 | 0.08 | 10.2 | 2.1e-12 | 0.95 |
| 1 | 0 | 424.3 | 1026.1 | 0.3e-6 | 98.5 | 0.73 | 0.92 | 7.85e-15 | 0.08 | 10.2 | 2.1e-12 | 0.92 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

Where:
- `frame`: Frame number
- `particle`: Unique particle identifier
- `x`, `y`: Coordinates in pixels
- `z`: Z-coordinate in meters (iteration 3 only)
- `size`: Particle size in nanometers
- `mass`: Brightness/intensity value (TrackPy uses "mass" for brightness)
- `brightness`: Normalized brightness (0-1)
- `raw_brightness`: Unnormalized brightness value (proportional to particle volume)
- `brightness_uncertainty`: Estimated uncertainty in brightness measurement
- `snr`: Signal-to-noise ratio estimate
- `diffusion_coefficient`: Theoretical diffusion coefficient based on particle size
- `focal_attenuation`: Brightness attenuation due to distance from focal plane (iteration 3 only)

## Directory Structure

- `src/`: Source code for all simulation versions
- `results/`: Output directory for simulation results
  - `iteration_1/`: Results from the basic 2D simulation
  - `iteration_2/`: Results from the enhanced 2D simulation
  - `iteration_3/`: Results from the 3D simulation with focal plane
  - `test_3_particles/`: Results from test simulations with 3 particles (2D)
  - `test_3_particles_3d/`: Results from 3D test simulations with 3 particles
  - `test_3_particles_gaussian/`: Results from Gaussian rendering tests

## Scripts

The project includes the following Python scripts:

1. **`simulation.py`**: Core simulation engine for iteration 1
2. **`simulation_v2.py`**: Enhanced simulation with varying particle sizes and track output (iteration 2)
3. **`simulation_v3.py`**: Advanced 3D simulation with focal plane effects (iteration 3)
4. **`simulation_v3_test_3p.py`**: Test version of the 3D simulation with only 3 particles
5. **`compare_simulations.py`**: Creates side-by-side comparisons of simulations at different temperatures
6. **`tracking_analysis.py`**: Analyzes particle trajectories and creates visualizations
7. **`track_utils.py`**: Utilities for working with particle track data
8. **`tif_converter.py`**: Utilities for converting between GIF and TIF formats

## Test Simulations

The project includes test simulations with only 3 particles that serve several purposes:

- Verify core simulation functionality with minimal particles
- Allow easier visual tracking of individual particles 
- Test simulation code changes quickly
- Generate sample data in same format as full simulation

For the 3D simulation, the test places particles at specific strategic positions:
- One particle near the focal plane
- One particle at medium distance from focal plane
- One particle far from focal plane

This arrangement helps visualize and verify the focal plane attenuation effects.

## Using Track Data with TrackPy

The CSV files can be loaded directly into TrackPy for analysis:

```python
import pandas as pd
import trackpy as tp

# Load our track data
tracks = pd.read_csv('simulation_v3_tracks.csv')

# Use TrackPy functions on the data
tp.plot_traj(tracks)
msd = tp.compute_msd(tracks)
```

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- Pandas
- Pillow (PIL)

## Future Enhancements

Potential future enhancements to the simulation could include:

1. **GPU Acceleration**: Improve performance for large numbers of particles
2. **Non-Spherical Particles**: Add support for rod-like or irregular particles
3. **Flow Field Simulation**: Add directed flow fields to the simulation
4. **Particle Interactions**: Model electrostatic or other interactions between particles
5. **Multiple Particle Types**: Simulate mixtures of different particle types
6. **Customizable Optics**: More detailed optical models with selectable objectives

## How to Cite

If you use this simulation in your research, please cite it as:

```
Nanoparticle Tracking Simulation. (2023). [Software]. https://github.com/username/nanoparticle_tracking
``` 