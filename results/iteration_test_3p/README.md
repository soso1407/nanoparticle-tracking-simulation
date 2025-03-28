# Nanoparticle Tracking Test Simulation (3 Particles)

This directory contains results from a test simulation with only 3 particles. This minimal version is designed for easier visual inspection and verification of the simulation functionality.

## Purpose

The test simulation serves several purposes:
- Verify the core simulation functionality with a minimal number of particles
- Allow for easier visual tracking of individual particles
- Provide a quick way to test changes to the simulation code
- Generate sample data in the same format as the full simulation

## Simulation Parameters

The test simulation uses the same parameters as the full simulation:
- Temperature: 298.15 K (room temperature)
- Viscosity: 1.0e-3 Pa·s (water)
- Mean particle radius: 50 nm
- Standard deviation of particle radius: 4.5 nm
- Frame size: 2048 x 2048 pixels
- Pixel size: 100 nm
- Diffusion time: 0.01 seconds
- Gaussian blur sigma: 2.0 pixels

## Output Files

- **simulation_v2_test_3p.gif**: Animated visualization of the 3-particle simulation
- **simulation_v2_test_3p.tif**: 16-bit TIF format for scientific image analysis
- **simulation_v2_test_3p_tracks.json**: Complete track data in JSON format
- **simulation_v2_test_3p_tracks.csv**: Track data in CSV format (compatible with TrackPy)
- **simulation_v2_test_3p_metadata.json**: Simulation parameters and metadata
- **particle_size_distribution.png**: Distribution of the 3 particle sizes
- **msd_by_size.png**: Mean Squared Displacement analysis (limited value with only 3 particles)

## Particle Details

The console output of the simulation provides detailed information about each of the 3 particles, including:
- Size (diameter in nm)
- Diffusion coefficient (m²/s)
- Brightness value
- Signal-to-noise ratio (SNR)
- Brightness uncertainty

## How to Run

To run this test simulation:

```bash
python src/simulation_v2_test_3p.py
```

## Notes

- With only 3 particles, statistical analyses like MSD by size category may not be meaningful
- The 3-particle simulation is meant for visual inspection and code verification, not for scientific analysis
- The random seed is not fixed, so each run will generate different particle properties and trajectories 