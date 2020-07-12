# Description: Read, extract and reformat data components
# Sections:
#   A: Read in image and annotation files
#   B: Formulate camera properties  

# SECTION A: Read in image and annotation files  

import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

def read_data(basepath):
    data = read_json(basepath)
    col_img, col_bytes = read_jpeg(basepath)
    ste_arr = read_stencil(basepath)
    return data, col_img, col_bytes, ste_arr

def read_json(basepath):
    file = basepath + ".json"
    with open(file, "r") as a:
        data = json.load(a)
    return data

def read_jpeg(basepath):
    file = basepath + "_colour.jpeg"    
    if not os.path.isfile(file):
        tif_file = basepath + "_colour.tif"
        tif_to_jpeg(tif_file)
    
    col_img = Image.open(file)
    col_bytes = tf.io.gfile.GFile(file, 'rb').read()
    return col_img, col_bytes

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
    print(image_path_jpeg)

def read_stencil(basepath):
    file = basepath + "_stencil.tif"     
    ste_img = Image.open(file, "r")
    ste_arr = np.array(ste_img, dtype = np.uint8)    
    return ste_arr

# SECTION B: Formulate camera properties

import numpy as np 
from math import tan

class Camera(object):

    def __init__(self, camera):

        self.pos = np.array(camera["location"], dtype=np.float64)
        self.rot = np.radians(camera["rotation"], dtype=np.float64)
        self.yaw = self.rot[2]
        
        self.C = np.array(camera["C"]["Values"], dtype=np.float64).reshape((4, 4)).T
        self.R = Camera.ReformatC(self.C)
        
        self.w = camera["screenW"]
        self.h = camera["screenH"]

        vfov = np.radians(camera["vfov"], dtype=np.float64)        
        self.fy = 1/tan(vfov/2)
        self.fx = self.h/self.w * self.fy
 
    @staticmethod
    def ReformatC(C):
        R_raw = C[1:, :3]
        r_norm = [x/np.linalg.norm(x) for x in R_raw]
        R = np.vstack(r_norm).T
        return R


        


