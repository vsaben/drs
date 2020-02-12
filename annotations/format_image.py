from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# A: Convert Tif to Jpeg

def TifToJpeg(image_path, resize_dims = None):
    
    # Function: Converts a single Tif file to Jpeg and removes the corresponding Tif file
    # Output: New Jpeg file directory
    
    # i: Convert - Image to array

    image_raw = Image.open(image_path)
    image_array = np.array(image_raw)
    image_rgb = image_array[:, :, 0:3]        
    if resize: 
        resized_image = Image.fromarray(image_rgb).resize(resize_dims)
        image_rgb = np.array(resized_image)

    # ii: Convert - Array to jpeg

    image_path_jpeg = image_path[:-3] + "jpeg"        
    plt.imsave(image_path_jpeg, image_rgb)
    print(image_path)
    os.remove(image_path)

    return image_path_jpeg
