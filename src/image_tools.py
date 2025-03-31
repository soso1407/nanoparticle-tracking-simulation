import numpy as np
from PIL import Image
import tifffile


# conversion and upscaling functions
def convert_png(source, output, scale=True):
    # open image
    img = Image.open(source)

    # convert to grayscale 16bit
    img = img.convert("I;16") 
    
    # convert to numpy array to scale to fill 16 bit range
    # currently using a linear scaler, but could implement a different function
    arr = np.array(img, dtype=np.uint16) * 257
    img_16bit = Image.fromarray(arr, mode="I;16")
    
    # save in tif format
    img_16bit.save(output, format="TIFF")

def scale_tif(source, output):
    # scale unscaled 16 bit tif image to fill entire 16 bit scale
    # convert to 16bit greyscale if not already set

    # open the image
    img = Image.open(source)

    # convert to 16bit if needed
    if img.mode != "I;16":
        print("converting", source, "to 16bit")
        img = img.convert("I;16")

    # get maximum pixel value
    max_pixel = np.max(np.array(img, dtype=np.uint16))
    if max_pixel < 256:

        print("Scaling", source, "to fill 16bits")
        
        arr = np.array(img, dtype=np.uint16) * 257
        img_16bit = Image.fromarray(arr, mode="I;16")

        img_16bit.save(output, format="TIFF")

    else:
        print("no scaling needed for", source)
        print("max pixel:", max_pixel)


def img_info(source, percentiles=False):
    # Print size and distribution info for image

    print("Info for", source, ":") 
    
    img = Image.open(source)

    print("Size:", img.size)
    print("Mode:", img.mode)

    max_pixel = np.max(np.array(img))
    min_pixel = np.min(np.array(img))
    avg_pixel = np.mean(np.array(img))

    print("max pixel:", max_pixel)
    print("min pixel:", min_pixel)
    print("average pixel:", avg_pixel)

    if percentiles:
        ps = [0, 25, 50, 75, 90, 95, 99, 99.999, 99.99999, 99.9999999]
        # Compute percentile values
        percentile_values = np.percentile(np.array(img, dtype=np.uint16), ps)
        
        # Print results
        for p, v in zip(ps, percentile_values):
            print(f"{p}th percentile: {v}")

    


def check_imgs_folder(folder, extension=None, percentiles=False):

    if extension:
        # only test images with extension
        files = glob.glob(f"{folder}/*.{extension}")

        for fname in files:
            img_info(fname, percentiles=percentiles)
            print()

    else:
        # test all image files
        exts = Image.registered_extensions()

        for ext in exts:
                                                # remove dot
            check_imgs_folder(folder, extension=ext[1:], percentiles=percentiles)