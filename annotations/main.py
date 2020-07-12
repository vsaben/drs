# Description: Compute multiple image annotations
# Function:
#   A. Move/sort file components
#   B. Compute multiple annotations

PKG_DIR = "C:\\Users\\Vaughn\\projects\\work\\drs\\annotations"

import sys
sys.path += [PKG_DIR]

# A: Move/sort file components

from pathlib import Path
import shutil
import os

def move_files(input_path, output_path, regex):
    
    files = [str(d) for d in Path(input_path).glob(regex)]
    
    for old_path in files:
        name = str(old_path)[(old_path.rfind("\\") + 1):]
        new_path = os.path.join(str(output_path), name)
        shutil.move(old_path, new_path)
        print(new_path)

# B: Compute multiple annotations


DB_DIR =  "C:\\Users\\Vaughn\\projects\\work\\drs\\annotations\\sample_data\\"
names = ["0113{:s}W00RD".format(str(d)) for d in range(72, 76 + 1)]

from image import Annotate

basepaths = [DB_DIR + d for d in names]
[Annotate(d, save_image = True, save_tfrecord = True) for d in basepaths]






'''
import sys

dir = "C:\\Users\\Vaughn\\projects\\work\\drs\\annotations"
sample_path = "\\sample_data" 

sys.path += [dir, dir + sample_path]

from format_image import tif_to_jpeg
from data import read_data
import numpy as np
np.set_printoptions(suppress=True)

# 1: Read in Data and Image

filename = "\\011317W00RD"
file = dir + sample_path + filename
data, col_img, ste_arr = read_data(file)

from data import Camera
camera = Camera(data["camerast"])

target = data["targetst"][0]

W = np.array(target["position"]["location"], dtype = np.float64)
tar_rot = np.radians(target["position"]["rotation"], dtype = np.float64)
bbox_max = np.array(target["position"]["bbox_max"], dtype = np.float64)

def GetDirectionalVectors(position):
        f = np.array(position["forward_vec"], np.float64)
        r = np.array(position["right_vec"], np.float64)
        u = np.array(position["up_vec"], np.float64)
        return f, r, u

f, r, u = GetDirectionalVectors(target["position"])
Wm = W + r * bbox_max[0] + f * bbox_max[1] + u * bbox_max[2] 
cWm = camera.R.T @ (Wm - camera.pos)

from math import sin, cos
def create_cam_rot_matrix(rot):
    x, y, z = np.radians(rot)

    Rx = np.array([
        [1, 0, 0],
        [0, sin(x), cos(x)],
        [0, cos(x), -sin(x)]
    ], dtype=np.float)
    Ry = np.array([
        [cos(y), 0, -sin(y)],
        [0, 1, 0],
        [sin(y), 0, cos(y)]
    ], dtype=np.float)
    Rz = np.array([
        [cos(z), sin(z), 0],
        [sin(z), -cos(z), 0],
        [0, 0, 1]
    ], dtype=np.float)
    return Rx @ Ry @ Rz

R = create_cam_rot_matrix(camera.rot)

P = np.array([ 1.0805524395165576, 1.9029209122487772E-08, -4.95299613093773E-13, 4.2120250617388514E-17, 
              2.0363960618603957E-09, 1.9209820988846258, 6.3865115995133392E-13, 0.0, 
              -3.631956087558799E-09, -4.92842655575032E-09, 2.3245810955354056E-05, 1.0, 
              -1.0146761809437521E-05, 5.01480289472056E-06, 0.15000348584112166, 0.0 ], dtype =np.float64).reshape((4, 4)).T






cX, cY, cZ, cw = P @ np.append(cWm, 1)

cX/cw
cY/cw


cam_pos = np.array(camera["location"], dtype = np.float64)
cam_rot = np.radians(camera["rotation"], dtype = np.float64)

from math import sin, cos, tan

def create_rot(euler):
    x, y, z = euler

    Rx = np.array([[1, 0, 0],
                    [0, cos(x), -sin(x)],
                    [0, sin(x), cos(x)]], dtype=np.float)
    Ry = np.array([[cos(y), 0, sin(y)],
                    [0, 1, 0],
                    [-sin(y), 0, cos(y)]], dtype=np.float)
    Rz = np.array([[cos(z), -sin(z), 0],
                    [sin(z), cos(z), 0],
                    [0, 0, 1]], dtype=np.float)
    R = Rz @ Ry @ Rx
    # R = np.hstack(R[0], R[2], R[1])
    return R

# 1: World co-ordinates


W # W
 # Wm

R_cam = create_rot(cam_rot)
R_cam_sw = np.vstack((R_cam[:, 0], R_cam[:, 2], -R_cam[:, 1])) # +/+/-

R_tar = create_rot(tar_rot)
R_tar @ bbox_max + W                                    # Wm 

cW = R_cam_sw @ (W - cam_pos)
-cW

cM = R_cam_sw @ R_tar @ bbox_max + cW
-cM

R = R_cam_sw @ R_tar 

tar_rel_rot = GetEulerFromR(R)
create_rot(tar_rel_rot) == R

fx = 1080/(1920*tan(np.radians(55)/2))
fy = 1/tan(np.radians(55)/2)

cW_x = fx * cW[0]/cW[2]
cW_y = fy * cW[1]/cW[2]

sW =[int(1920 * 0.5 *(1 + cW_x)), int(1080 * 0.5 * (1 - cW_y))]
sW

cM_x = fx * cM[0]/cM[2]
cM_y = fy * cM[1]/cM[2]

sM = [int(1920 * 0.5 *(1 + cM_x)), int(1080 * 0.5 * (1 - cM_y))]
sM

ShowBB(col_img, [sW, sM])

def ShowBB(col_img, points):

    col_arr = np.array(col_img)

    for p in points:
        cv2.circle(col_arr, tuple(p), 2, (255, 255, 255), -1)
    
    # cv2.drawContours(col_arr, [points], 0, (255, 0, 0))

    cv2.imshow('image', col_arr)
    cv2.waitKey(0)

# Using camera matrix

C = np.array(data["camerast"]["C"]["Values"]).reshape((4, 4)).T
W = np.array(data["targetst"][0]["position"]["location"])
gmax = np.array(data["targetst"][0]["position"]["bbox_max"])
gmin = np.array(data["targetst"][0]["position"]["bbox_min"])

def W2S(C, W):

    vFwd = C[3, :]
    vRight = C[1, :]
    vUp = C[2, :]

    Z = np.dot(vFwd[:3], W) + vFwd[3] 
    X = np.dot(vRight[:3], W) + vRight[3]
    Y = np.dot(vUp[:3], W) + vUp[3]V

    invw = 1 / Z
    X *= invw
    Y *= invw

    return [int(1920 * 0.5 *(1 + X)), int(1080 * 0.5 * (1 - Y))]

def W2S_2(C, W):

    X, Y, Z = C[1:, :3] @ W + C[1:, 3] 
    invW = 1 / Z
    
    X *= invW
    Y *= invW
        
    return [int(1920 * 0.5 *(1 + X)), int(1080 * 0.5 * (1 - Y))]





# 2: Find and visualise car center

def FPSView(pos, euler):

    pitch, _ , yaw = euler           # Assume: No camera roll

    cosP, sinP = cos(pitch), sin(pitch)
    cosY, sinY = cos(yaw), sin(yaw)

    xaxis = np.array([cosY, 0, -sinY])
    yaxis = np.array([sinY * sinP, cosP, cosY * sinP])
    zaxis = np.array([sinY * cosP, -sinP, cosP * cosY])

    V = np.zeros((4, 4))
    V[:3, 0] = xaxis
    V[:3, 1] = yaxis
    V[:3, 2] = zaxis
    V[3, :] = np.array([-np.dot(xaxis, pos), -np.dot(yaxis, pos), -np.dot(zaxis, pos), 1])

    return V

def GetV(euler, translation) -> np.matrix:
    rotation = create_rot(euler)
    x, y, z = translation
    
    translation_mtx = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [-x, -y, -z, 1]], dtype=np.float64)
    
    rot_mtx = np.hstack((rotation, [[0], [0], [0]]))
    rot_mtx = np.vstack((rot_mtx, [0, 0, 0, 1]))
    V =  rot_mtx.T @ translation_mtx
    return V

V = GetV(data["camerast"]["rotation"], data["camerast"]["location"])

def GetP(camera):

    W = camera["screenW"]
    H = camera["screenH"]
    vfov = np.radians(camera["vfov"])
    fclip = camera["fclip"]
    nclip = camera["nclip"]

    P = np.array([[H/(W*tan(vfov/2)), 0, 0, 0], 
                  [0, 1/tan(vfov/2), 0, 0], 
                  [0, 0, -(nclip + fclip)/(nclip - fclip), -fclip*nclip/(nclip - fclip)], 
                  [0, 0, 1, 0]])

    return P

P = GetP(data["camerast"])

def W2S(V, P, screenW, screenH, coords):

    ncol = coords.shape[1]
    coords = np.vstack((coords, [1] * ncol))
    clip = np.dot(P, np.dot(V, coords))
    ndc = clip/clip[3, :]

    T = np.array([[screenW/2, 0, 0, screenW/2],
                  [0, -screenH/2, 0, screenH/2]])

    res = np.dot(T, ndc).astype(int)
    return res



points_bb = []
for i in range(len(data["targetst"])):

    target = data["targetst"][i]

    pos_dict = target["position"]

    pos = np.array(pos_dict["location"])
    rot = np.radians(pos_dict["rotation"])
    f = np.array(pos_dict["forward_vec"])
    r = np.array(pos_dict["right_vec"])
    u = np.array(pos_dict["up_vec"])

    gmin = np.array(pos_dict["bbox_min"])
    gmax = np.array(pos_dict["bbox_max"])

    

    # corners = [[gmin[0] * r + gmax[1] * f + gmax[2] * u ], # FTL
    #           [gmax[0] * r + gmax[1] * f + gmax[2] * u ], # FTR
    #           [gmin[0] * r + gmax[1] * f + gmin[2] * u ], # FBL
    #           [gmax[0] * r + gmax[1] * f + gmin[2] * u ], # FBR
    #           [gmin[0] * r + gmin[1] * f + gmax[2] * u ], # BTL
    #           [gmax[0] * r + gmin[1] * f + gmax[2] * u ], # BTR
    #           [gmin[0] * r + gmin[1] * f + gmin[2] * u ], # BBL
    #           [gmax[0] * r + gmin[1] * f + gmin[2] * u ]] # BBR
    

    # bbox_dims = np.array(list(product(
    #                          [gmin[0], gmax[0]],
    #                          [gmin[1], gmax[1]],
    #                          [gmin[2], gmax[2]],
    #                           ))).T
    # R = euler_to_dcm(rot)

    # corners = np.dot(R, bbox_dims).T
    
    bbox_dict = target["bbox"]
    corners = [v for v in bbox_dict.values()]
    points_bb += corners 

    points_bb += [W2SFOR(V, P, pos + x) for x in corners]
'''

