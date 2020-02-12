from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# 1: Functions

def tif_to_jpeg(image_path, resize_dims = None):
    
    # Function: Converts a single Tif file to Jpeg and removes the corresponding Tif file
    # Output: New Jpeg file directory
    
    # i: Convert - Image to array

    image_raw = Image.open(image_path)
    image_array = np.array(image_raw)
    image_rgb = image_array[:, :, 0:3]        
    
    if resize_dims is not None: 
        resized_image = Image.fromarray(image_rgb).resize(resize_dims)
        image_rgb = np.array(resized_image)

    # ii: Convert - Array to jpeg

    image_path_jpeg = image_path[:-3] + "jpeg"        
    plt.imsave(image_path_jpeg, image_rgb)
    print(image_path)
    os.remove(image_path)

    return image_path_jpeg

def depth_to_usable_depth(depth_path):

    # Function: Makes original depth image visible
    # Check: Pixel linearisation

    im_depth = Image.open(depth_path, mode = "r")                                             # Open
    im_depth_pixel = list(im_depth.getdata())                                                 # Pixel information

    im_depth_epixel = [math.exp(x) for x in im_depth_pixel]                                   # Linearised pixels [Unlikely correct]
    im_depth_array = np.array(im_depth_epixel, dtype = "float32").reshape((720, 1280))        # Reshape array

    plt.imsave(depth_path, im_depth_array)                                                    # Save plot (overwrite)

def stencil_to_usable_stencil(stencil_path):

    # Function: Makes original stencil image visible
    # Check: Loss of per pixel flags 

    im_stencil = Image.open(stencil_path)                                                     # Open    
    im_stencil_array = np.array(im_stencil, dtype = "uint8").reshape((720, 1280))             # Reshape array

    plt.imsave(stencilpath, im_stencil_array)          