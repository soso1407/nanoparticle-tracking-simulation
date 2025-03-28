# Iteration 2: Enhanced Brightness Modeling and 16-bit TIF Output

This iteration implements several important enhancements to the nanoparticle tracking simulation:

## 1. Realistic Brightness Modeling

- **Area-Based Brightness**: Particle brightness now scales with the cross-sectional area (πr²), making larger particles appear brighter in proportion to their size.
- **SNR Calculation**: Signal-to-noise ratio (SNR) is calculated based on particle size, allowing for more realistic modeling of detection quality.
- **Brightness Uncertainty**: Each particle's brightness includes an uncertainty component, calculated as brightness/SNR.
- **Temporal Fluctuations**: Brightness values fluctuate realistically from frame to frame based on the calculated uncertainty.

## 2. 16-bit TIF Image Format

- **Higher Dynamic Range**: Images are now saved in 16-bit TIF format (0-65535 grayscale levels) instead of 8-bit (0-255), providing significantly higher dynamic range.
- **Scientific Analysis Support**: The 16-bit format better supports scientific image analysis by preserving subtle brightness variations.
- **Enhanced Brightness Resolution**: Particle brightness differences are more precisely represented with the extended dynamic range.
- **Implementation**: 8-bit pixel values (0-255) are scaled to the full 16-bit range (0-65535) using a factor of 257.

## 3. Enhanced Track Data

The simulation now includes enriched track data with the following fields:

- `raw_brightness`: Base brightness value calculated from particle size
- `brightness_uncertainty`: Estimated uncertainty in brightness measurement
- `snr`: Signal-to-noise ratio for each particle

## Output Files

- **simulation_v2.gif**: Animated visualization of the simulation
- **simulation_v2.tif**: 16-bit TIF format image sequence
- **simulation_v2_tracks.json**: Complete track data in JSON format
- **simulation_v2_tracks.csv**: Track data in CSV format (compatible with TrackPy)
- **simulation_v2_metadata.json**: Simulation parameters and metadata, including the 16-bit TIF depth

## Notes for Analysis

When analyzing the 16-bit TIF images:
- Ensure your software is configured to handle 16-bit data correctly
- Brightness values span 0-65535 instead of 0-255
- For comparison with 8-bit images, divide by 257 to convert back to 8-bit range

## Implementation Details

The 16-bit TIF conversion is implemented in two places:
1. Direct saving in `simulation_v2.py` during simulation
2. Conversion utility in `tif_converter.py` for processing existing GIF files

## File Formats

### Simulation Files

- `simulation_v2.gif` - Animation of the simulation
- `simulation_v2.tif` - TIFF version of the simulation
- `particle_size_distribution.png` - Histogram showing the distribution of particle sizes
- `msd_by_size.png` - Mean Squared Displacement plot for different particle size categories

### Track Data

- `simulation_v2_tracks.json` - Particle tracks in JSON format
- `simulation_v2_tracks.csv` - Particle tracks in CSV format (TrackPy compatible)
- `simulation_v2_metadata.json` - Metadata about the simulation

## Track Format

### JSON Format

The JSON track format uses the following structure:

```json
[
  {
    "particle_id": 0,
    "frames": [0, 1, 2, ...],
    "positions": [[x0, y0], [x1, y1], [x2, y2], ...],
    "sizes": [101.2, 101.2, 101.2, ...],
    "brightnesses": [0.8, 0.8, 0.8, ...],
    "raw_brightness": [7.85e-15, 7.85e-15, ...],
    "brightness_uncertainty": [0.08, 0.08, ...],
    "snr": [10.2, 10.2, ...],
    "diffusion_coefficient": [2.1e-12, 2.1e-12, ...]
  },
  {
    "particle_id": 1,
    ...
  }
]
```

### CSV Format (TrackPy Compatible)

The CSV format follows TrackPy's convention:

| frame | particle | x | y | size | mass | raw_brightness | brightness_uncertainty | snr | diffusion_coefficient |
|-------|----------|---|---|------|------|---------------|------------------------|-----|----------------------|
| 0 | 0 | 423.1 | 1024.7 | 98.5 | 0.75 | 7.85e-15 | 0.08 | 10.2 | 2.1e-12 |
| 1 | 0 | 424.3 | 1026.1 | 98.5 | 0.73 | 7.85e-15 | 0.08 | 10.2 | 2.1e-12 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 0 | 1 | 823.4 | 562.1 | 105.2 | 0.82 | 8.91e-15 | 0.05 | 15.8 | 1.9e-12 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

Where:
- `frame`: Frame number
- `particle`: Unique particle identifier
- `x`, `y`: Coordinates in pixels
- `size`: Particle size in nanometers
- `mass`: Brightness/intensity value (TrackPy uses "mass" for brightness)
- `raw_brightness`: Unnormalized brightness value (proportional to particle area)
- `brightness_uncertainty`: Estimated uncertainty in brightness measurement
- `snr`: Signal-to-noise ratio estimate
- `diffusion_coefficient`: Theoretical diffusion coefficient based on particle size

## Brightness Modeling

The brightness characteristics are modeled as follows:

1. **Base brightness** is proportional to the particle's cross-sectional area (πr²).
2. **Signal-to-noise ratio (SNR)** is higher for larger particles, with some random variation.
3. **Brightness uncertainty** is calculated as brightness/SNR.
4. **Fluctuating brightness** incorporates random noise based on the uncertainty.

This realistic modeling of brightness is important for:
- Simulating how real particle tracking algorithms would perform
- Testing detection and tracking algorithms with varying signal quality
- Understanding the relationship between particle size and detection confidence

## Using Track Data with TrackPy

The CSV files can be loaded directly into TrackPy for analysis. For example:

```python
import pandas as pd
import trackpy as tp

# Load our track data
tracks = pd.read_csv('simulation_v2_tracks.csv')

# Use TrackPy functions on the data
tp.plot_traj(tracks)
msd = tp.compute_msd(tracks)
```

## Converting Between Formats

The `track_utils.py` module provides utilities for converting between different track formats:

```python
from src.track_utils import TrackConverter

# Convert CSV to JSON
tracks_json = TrackConverter.csv_to_json('tracks.csv', 'tracks.json')

# Convert JSON to DataFrame
df = TrackConverter.json_to_trackpy('tracks.json')
``` 