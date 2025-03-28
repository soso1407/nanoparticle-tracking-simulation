# Nanoparticle Tracking Test Simulation (3 Particles with Direct Gaussian Rendering)

This directory contains results from a test simulation with only 3 particles, using physically accurate direct Gaussian rendering. This enhanced version creates more realistic particle images by directly modeling particles as 2D Gaussians with peak intensities proportional to particle volume.

## Improved Physical Model

The main improvements in this version:

1. **Direct Gaussian Rendering**: Particles are directly rendered as 2D Gaussians with appropriate peak intensities, rather than drawing circles and then applying a global Gaussian blur.

2. **Volume-Based Brightness**: Particle brightness is now proportional to particle volume (r³), which is more physically accurate for scattering intensity in optical microscopy.

3. **Diffraction-Limited Size**: The width of the Gaussian is scaled based on particle size, reflecting how larger particles appear slightly larger in diffraction-limited microscopy.

4. **Background Noise**: Realistic background noise is added to the image.

## Purpose

The simulation serves to:
- Test the new physically accurate Gaussian rendering
- Allow visual comparison between the circle+blur method and direct Gaussian rendering
- Verify that brightness scales correctly with particle volume
- Provide a minimal testbed for future optical physics improvements

## Simulation Parameters

The simulation uses the same physical parameters as other versions:
- Temperature: 298.15 K (room temperature)
- Viscosity: 1.0e-3 Pa·s (water)
- Mean particle radius: 50 nm
- Standard deviation of particle radius: 4.5 nm
- Frame size: 2048 x 2048 pixels
- Pixel size: 100 nm
- Diffusion time: 0.01 seconds
- Gaussian sigma: 2.0 pixels (baseline for a 100nm particle)

## Output Files

- **simulation_v2_test_3p_gaussian.gif**: Animated visualization of the 3-particle simulation
- **simulation_v2_test_3p_gaussian.tif**: 16-bit TIF format for scientific image analysis
- **simulation_v2_test_3p_gaussian_tracks.json**: Complete track data in JSON format
- **simulation_v2_test_3p_gaussian_tracks.csv**: Track data in CSV format (compatible with TrackPy)
- **simulation_v2_test_3p_gaussian_metadata.json**: Simulation parameters and metadata
- **particle_size_distribution.png**: Distribution of the 3 particle sizes
- **msd_by_size.png**: Mean Squared Displacement analysis

## Particle Details

The console output of the simulation provides detailed information about each particle:
- Size (diameter in nm)
- Diffusion coefficient (m²/s)
- Normalized brightness value (0-1)
- Raw brightness (volume-based)
- Signal-to-noise ratio (SNR)
- Brightness uncertainty

## How to Run

To run this improved Gaussian test simulation:

```bash
python src/simulation_v2_test_3p.py
```

## Physical Basis for Volume-Based Brightness

In optical microscopy of nanoparticles:
1. For particles much smaller than the wavelength of light (Rayleigh scattering regime), scattering intensity is proportional to particle volume squared (r⁶)
2. For particles comparable to the wavelength of light (Mie scattering regime), scattering is approximately proportional to particle volume (r³)

This simulation uses a proportionality to volume (r³) as a reasonable approximation for nanoparticles in the 80-120nm range.

## Notes

- This test version uses only 3 particles to make visual inspection and verification easier
- The Gaussian rendering approach is more computationally intensive but produces more realistic particle images
- The volume-based brightness calculation is physically more accurate than the previous area-based calculation 