# TrackPy Analysis

This directory contains the results of analyzing the nanoparticle tracking data using the TrackPy library.

## Getting Started

To run the TrackPy analysis, you'll need to install TrackPy:

```bash
pip install trackpy
```

Then you can run the analysis script:

```bash
python src/trackpy_example.py
```

## Expected Output

The analysis generates several visualizations:

1. **trackpy_trajectories.png**: Visualization of all particle trajectories
2. **trackpy_msd.png**: Mean Squared Displacement of individual particles 
3. **trackpy_ensemble_msd.png**: Ensemble Mean Squared Displacement
4. **trackpy_size_trajectories.png**: Trajectories broken down by particle size
5. **trackpy_size_msd.png**: Mean Squared Displacement by particle size category
6. **trackpy_size_vs_brightness.png**: Analysis of the relationship between particle size and brightness

## TrackPy Features Used

The example demonstrates several key TrackPy features:

- `tp.plot_traj()`: Visualizing particle trajectories
- `tp.imsd()`: Computing individual Mean Squared Displacement
- `tp.emsd()`: Computing ensemble Mean Squared Displacement

## Custom Analysis

The script also demonstrates custom analysis:

- Size-based categorization and comparison
- Relationship between particle size and brightness
- Fitting of diffusion curves

## Integration with Other Libraries

TrackPy works well with pandas and numpy, making it easy to perform additional analyses on the tracking data. 