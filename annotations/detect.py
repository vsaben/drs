"""
    Description: Projects annotations into screen co-ordinates
    Function:
    - PART A: Get all 3D bounding boxes co-ordinates 
              (in and off screen)
    - PART B: Extract 3D bounding box co-ordinates (on screen) 
              from PART A
    - PART C: Extract 2D bounding boxes from PART A
"""

import numpy as np

import utils

# PART A: All 3D BB ==============================================

def get_all_3d(camera, tar_pos, tar_dims, tar_rot):
    
    """Gets all 3D BB (on and off screen) screen co-ordinates

    :note: (-1, -1) returned if coord off screen

    :param camera: camera class with rotation and projection 
                   properties
    :param tar_pos: target center position rel. to the camera 
                    (in meters)
    :param tar_dims: target half-dimensions from tar_pos
    :param tar_rot: target rotation (yaw rel. to camera)

    :result: all 3d bb screen co-ordinates (dictionary)
    """

    abs_tar_rot = tar_rot.copy()
    abs_tar_rot[2] = utils.get_rel_yaw(camera.rot[2], abs_tar_rot[2], 
                                       is_pos_axis = (camera.R[2, 1] >= 0), 
                                       is_rel_cam = False)
    R_tar = utils.create_target_rot(abs_tar_rot)

    corner_dims_dict = utils.get_corner_dim(tar_dims)
    corner_dims = list(corner_dims_dict.values())
    corner_dims = np.vstack(corner_dims).T   
    
    corner_cam = (camera.R.T @ R_tar @ corner_dims).T + tar_pos  
    return {k: utils.C2S(camera, c) for k, c in zip(corner_dims_dict.keys(), corner_cam)}

# PART B/C: Extract on-screen 3D and 2D BB from PART A ===============

def get_3d(camera, all_bb3d):
    
    """Extract 3D BB dictionary of on-screen co-ordinates""" 

    bb3d_onscreen = {k: v for k, v in all_bb3d.items() 
                     if utils.coordinate_on_screen(camera, v)}
    return bb3d_onscreen
                  
def get_2d(camera, all_bb3d):
    
    """Extract 2D BB (xmin, ymin, xmax, ymax)"""

    bb3d = tuple(all_bb3d.values())
    bb3d = np.vstack(bb3d)

    x_vals = bb3d[:, 0]
    y_vals = bb3d[:, 1]

    xmin = np.min(x_vals)
    xmax = np.max(x_vals)
    ymin = np.min(y_vals)
    ymax = np.max(y_vals)

    if xmin < 0: xmin = 0
    if xmax > (camera.w - 1): xmax = camera.w - 1
    if ymin < 0: ymin = 0
    if ymax > (camera.h - 1): ymax = camera.h - 1

    return [xmin, ymin, xmax, ymax]
