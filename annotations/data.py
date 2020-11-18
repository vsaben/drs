"""
    Description: Read, extract and partially reformat data components.
                 Delete invalid tests.   
"""

from files import move_files

from PIL import Image
from pathlib import Path

import json
import os
import shutil
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
        print("{s}: {s} moved to error folder".format(ERROR_DICT['camera'], 
                                                      basepath))
        return [False]*6  

    col_img, col_bytes = read_jpeg(basepath)
    ste_arr = read_stencil(basepath)
    return True, data, control, col_img, col_bytes, ste_arr

# Error assessment

ERROR_DICT = {'camera': 'Camera below ground'}

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

    tif_path = basepath + "_colour.tif"
    image_path = basepath + "_colour.jpeg" 

    # Convert: Image to array

    image_raw = Image.open(tif_path)
    image_array = np.array(image_raw)
    image_rgb = image_array[:, :, 0:3]        
    
    if resize_dims is not None: 
        resized_image = Image.fromarray(image_rgb).resize(resize_dims)
        image_rgb = np.array(resized_image)

    # Convert: Array to jpeg
           
    plt.imsave(image_path, image_rgb)
    print(image_path)

def read_stencil(basepath):

    """Read in and save (if not exist) stencil array"""

    file = basepath + "_stencil.tif"     
    ste_img = Image.open(file, "r")
    ste_arr = np.array(ste_img, dtype = np.uint8)
    msk_arr = np.isin(ste_arr, [2, 130])

    ste_path = basepath + "_stencil.jpeg"
    if not os.path.isfile(ste_path):
        plt.imsave(ste_path, msk_arr, cmap=cm.gray)
        print(ste_path)
    return msk_arr

def move_test_files(files, folder, levelup = 0):

    """Move all files associated with a test"""
 
    parent = Path(files[0]).parents[levelup]
    out_dir = os.path.join(parent, folder) 
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    move_files(files, out_dir, False)







