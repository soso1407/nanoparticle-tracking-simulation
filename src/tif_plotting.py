import numpy as np
from PIL import Image
import tifffile
import matplotlib.pyplot as plt


# function to plot tif file
def plot_tif(source, normalize = False, frame=None, subset=None):

    image =  None
    
    if frame == None:
        image = tifffile.imread(source)
    else:
        with tifffile.TiffFile(source) as tif:
            image = tif.pages[frame].asarray()

    if subset != None:
        x_start, x_end = subset[0]
        y_start, y_end = subset[1]
        image = image[y_start:y_end, x_start:x_end]
        
    if (normalize):
        image = image.astype(np.float32)
        image -= image.min()  # Shift min to 0
        image /= image.max()  # Scale max to 1

    

    # Generate X, Y coordinates
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    X, Y = np.meshgrid(x, y)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    ax.plot_surface(X, Y, image, cmap='gray', edgecolor='none')
    
    # Labels and view angle
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Brightness')
    ax.set_title('3D Heightmap of Grayscale TIFF')
    ax.view_init(elev=60, azim=225)  # Adjust view angle
    
    plt.show()