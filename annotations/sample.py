"""
    Description: Recollect dense annotations for a single target. 
    Testing environment for raw annotation interpretation and formatting.  

    Function:
    - PART A: Read in data, image from the command line
    - PART B: Extract target (manually)
    - PART C: Display dense annotations
"""

import argparse
import numpy as np
import math
import cv2

from data import read_data
from target import Target
import utils 

parser = argparse.ArgumentParser()
parser.add_argument("basepath", type=str, help="relative sample basepath")
parser.add_argument("target_id", type=int, help="unique target identification number")
args = parser.parse_args()

# PART A: Read in data, image from the command line ==========================

isvalid, data, control, col_img, col_bytes, ste_arr = read_data(args.basepath)

# PART B: Extract targets (manually) =========================================

if isvalid:

    target = data["targetst"][args.target_id]

    # Position

    target_pos = target["position"]

    W_ud = np.array(target_pos["location"], np.float64)                               # Center (Unequal dimensions)
    print("W_ud: ", W_ud)

    dim_min = np.array(target_pos["bbox_min"], np.float64)
    dim_max = np.array(target_pos["bbox_max"], np.float64)
    dim_off = dim_min + dim_max
    print("dim_off: ", dim_off)

    f = np.array(target_pos["forward_vec"], np.float64)
    r = np.array(target_pos["right_vec"], np.float64)
    u = np.array(target_pos["up_vec"], np.float64)

    W = W_ud + r * dim_off[0] + f * dim_off[1] + u * dim_off[2] # Center (Equal dimensions)
    print("W: ", W)

    camera = data["camerast"]  

    # Dimensions

    hdims = (dim_max - dim_min) / 2 
    print("hdims: ", hdims)
    
    # Camera matrix

    #C = np.array(camera["C"]["Values"], dtype=np.float64).reshape((4, 4)).T
    #R_raw = C[1:, :3]
    #R_norm = [x/np.linalg.norm(x) for x in R_raw]
    #R_cam = np.vstack(R_norm).T
    #print("R_cam (C): ", R_cam)

    x, y, z = cam_rot = np.radians(camera["rotation"])

    Rx = np.array([[1, 0, 0],
                   [0, math.cos(x), -math.sin(x)],
                   [0, math.sin(x), math.cos(x)]], dtype=np.float64)
    Ry = np.array([[math.cos(y), 0, math.sin(y)],
                   [0, 1, 0],
                   [-math.sin(y), 0, math.cos(y)]], dtype=np.float64)
    Rz = np.array([[math.cos(z), -math.sin(z), 0],
                   [math.sin(z), math.cos(z), 0],
                   [0, 0, 1]], dtype=np.float64)

    ISwitch = np.array([[1, 0, 0], 
                        [0, 0, 1], 
                        [0, 1, 0]], dtype=np.float64)

    R_zxy = Rz @ Rx @ Ry
    R_cam = R_zxy @ ISwitch
    print('R_cam (switch): ', R_cam)

    # Target/Camera yaw 

    print("cam_rot w\ yaw orientation indicator: ", cam_rot, R_cam[2, 1])

    tar_rot_abs = np.radians(target_pos['rotation'], dtype=np.float64)
    print('tar_rot_abs', tar_rot_abs)

    rel_yaw = tar_rot_abs[2] - cam_rot[2]
    print('rel_yaw (pre)', np.degrees(rel_yaw))


    camera_pos = np.array(camera["location"], np.float64)
    cW_center = np.matmul(R_cam.T, W - camera_pos)              # Target position relative to the camera 
    print("cW_center: ", cW_center)

    

    if R_cam[2, 1] < 0: 
        if -np.pi <= rel_yaw <= 0: 
            rel_yaw += np.pi
            print('-180 <> 0')
        elif 0 < rel_yaw <= np.pi: 
            rel_yaw -= np.pi
            print('0 <> 180')
    
    print('rel_yaw', np.degrees(rel_yaw))
    

    R_tar = utils.create_target_rot(tar_rot_abs)
    print("R_tar: ", R_tar)

    R = np.matmul(R_cam.T, R_tar).T

    x, y, z = hdims
    corner_dims = np.array([[-x, y, z], 
                            [x, y, z],    
                            [-x, y, -z],  
                            [x, y, -z],   
                            [-x, -y, z],  
                            [x, -y, z],   
                            [-x, -y, -z], 
                            [x, -y, -z]]).T 

    cW_dims = np.matmul(R.T, corner_dims).T + cW_center
    cW = np.vstack((cW_dims, cW_center))

    # PART C: Display dense annotations ===========================================

    clipW = (cW[:, :2].T / cW[:, 2]).T

    vfov = np.radians(camera["vfov"], dtype=np.float64)
    HEIGHT = camera["screenH"]
    WIDTH = camera["screenW"]

    camera_fy = 1/math.tan(vfov/2)
    camera_fx = HEIGHT/WIDTH * camera_fy

    proj_factors = np.array([[camera_fx, 0], 
                             [0, camera_fy]])
     
    ndcW = np.matmul(clipW, proj_factors)
    sW = [(int(round((WIDTH - 1) * 0.5 * (1 + ndc_x))), 
           int(round((HEIGHT - 1) * 0.5 * (1 - ndc_y)))) for ndc_x, ndc_y in ndcW]

    col_arr = np.array(col_img)
    for p in sW:
        cv2.circle(col_arr, tuple(p), 2, (255, 255, 255), -1)

    bbox = [v['Item2'] for k, v in target["bbox"].items()]
    for b in bbox: 
        cv2.circle(col_arr, tuple(b), 2, (0, 0, 255), -1)
    
    print("sW: ", sW)    
    print("bbox: ", bbox)

    cv2.imshow('image', col_arr)
    cv2.waitKey(0)
    

    


