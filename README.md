# Nanoparticle Tracking Simulation

A Python-based simulation of nanoparticle tracking with both 2D and 3D implementations. This project simulates the Brownian motion of nanoparticles and includes realistic physics-based calculations for particle movement, brightness, and imaging effects.

## Features

- 2D and 3D particle tracking simulation
- Physics-based Brownian motion using Stokes-Einstein equation
- Realistic particle brightness calculations based on volume
- Signal-to-noise ratio (SNR) calculations
- Focal plane attenuation in 3D
- Camera noise simulation
- Particle size variation
- Track data export compatible with TrackPy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/nanoparticle-tracking.git
cd nanoparticle-tracking
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install numpy pandas matplotlib pillow
```

## Usage

The simulation includes multiple versions with increasing complexity:

1. Basic 2D simulation:
```python
from simulation.simulation import NanoparticleSimulator

simulator = NanoparticleSimulator()
frames = simulator.run_simulation(num_frames=100)
```

2. Enhanced 2D simulation with variable particle sizes:
```python
from simulation.simulation_v2 import NanoparticleSimulatorV2

simulator = NanoparticleSimulatorV2()
frames = simulator.run_simulation(num_frames=100)
```

3. Full 3D simulation with focal plane effects:
```python
from simulation.simulation_v3 import NanoparticleSimulator3D

simulator = NanoparticleSimulator3D()
frames = simulator.run_simulation(num_frames=100)
```

## Parameters

Key simulation parameters include:

- `temperature`: Temperature in Kelvin (default: 298.15 K)
- `viscosity`: Fluid viscosity in Pa·s (default: 1.0e-3 Pa·s for water)
- `particle_radius`: Particle radius in meters (default: 50e-9 m)
- `frame_size`: Frame size in pixels (default: 2048x2048)
- `pixel_size`: Physical size of one pixel (default: 100e-9 m)
- `diffusion_time`: Time step between frames (default: 1.0 s)

For 3D simulation additional parameters:
- `z_range`: Range of z positions in meters
- `focal_plane`: Z position of focal plane
- `depth_of_field`: Depth of field in meters

## Physics

The simulation uses several physical principles:

1. **Brownian Motion**: Calculated using the Stokes-Einstein equation:
   D = k_B * T / (6 * π * η * r)
   where:
   - D: Diffusion coefficient
   - k_B: Boltzmann constant
   - T: Temperature
   - η: Viscosity
   - r: Particle radius

2. **Brightness Calculation**: Based on particle volume (r³)

3. **Signal-to-Noise Ratio**: Scales with particle size

4. **Focal Attenuation**: Uses a modified Gaussian function based on distance from focal plane

## License

MIT License 