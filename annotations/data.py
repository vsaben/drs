"""
    Description: Read, extract and partially reformat data components.
                 Delete invalid tests.   
"""

from PIL import Image

import json
import os
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf

def read_data(basepath):

    """Read in raw annotation json, convert .tif to .jpeg and decipher
    stencil array

    :param basepath: common file path for a single simulation's elements

    :result isvalid: whether image is valid (positive camera altitude check)  
    :result data: all-encompassing data dictionary 
    :result control: control sub-dictionary
    :result col_img: colour 'Image' object 
    :result col_bytes: colour 'bytes' object
    :result ste_arr: stencil array
    """

    data = read_json(basepath)
    control = data["controlst"]
    
    # Check: Camera altitude 
    
    cam_altitude = int(control["assess_factors"]["altitude"]) 
    if cam_altitude <= 0: 
        del data
        remove_test_files(basepath)
        print("Camera below ground, {:s} deleted".format(basepath))
        return [False]*6  

    col_img, col_bytes = read_jpeg(basepath)
    ste_arr = read_stencil(basepath)
    return True, data, control, col_img, col_bytes, ste_arr

def read_json(basepath):
    
    """Read simulation data .json file"""

    file = basepath + ".json"
    with open(file, "r") as a:
        data = json.load(a)
    return data

def read_jpeg(basepath):
    
    """Read in colour image in 'Image' and 'bytes' format"""
    
    file = basepath + "_colour.jpeg"    
    if not os.path.isfile(file): 
        tif_to_jpeg(basepath)
    
    col_img = Image.open(file)
    col_bytes = tf.io.gfile.GFile(file, 'rb').read()
    return col_img, col_bytes

def tif_to_jpeg(basepath, resize_dims = None):
    
    """Replaces and reformats colour image .tif with .jpeg. Resizes image
    dimensions if specified
    
    :param basepath: full path to colour image .tif file
    :option resize_dims: target image dimensions
    
    :result: saved colour image .jpeg file (same directory)
    """

    tif_file = basepath + "_colour.tif"

    # Convert: Image to array

    image_raw = Image.open(tif_file)
    image_array = np.array(image_raw)
    image_rgb = image_array[:, :, 0:3]        
    
    if resize_dims is not None: 
        resized_image = Image.fromarray(image_rgb).resize(resize_dims)
        image_rgb = np.array(resized_image)

    # Convert: Array to jpeg

    image_path_jpeg = image_path[:-3] + "jpeg"        
    plt.imsave(image_path_jpeg, image_rgb)
    print(image_path_jpeg)

def read_stencil(basepath):

    """Read in stencil array"""

    file = basepath + "_stencil.tif"     
    ste_img = Image.open(file, "r")
    ste_arr = np.array(ste_img, dtype = np.uint8)    
    return ste_arr

def remove_test_files(basepath):

    """Remove all files associated with a test"""

    split_ind = basepath.rfind("\\") + 1 
    base = basepath[:split_ind]
    test = basepath[split_ind:]
    
    test_files = [os.remove(d) for d in Path(base).glob("{}*".format(test))]
    print("Deleted: ", test)







