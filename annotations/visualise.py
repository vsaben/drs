"""
    Description: Superimpose and/or save image annotations

    Options: (or combinations thereof)
        
        1. 3D bbox
        2. 2D bbox
        3. Features
"""

import cv2
import numpy as np
import random

def draw_annotations(basepath, col_img, control, environment, targets, camera, 
                     setting=2, 
                     display_features = True, 
                     save_image = False):
 
    _2d = 1
    _3d = 2
    all = 3

    col_arr = cv2.cvtColor(np.array(col_img), cv2.COLOR_RGB2BGR)

    isday = control["assess_factors"]["timeofday"] == 'd'
    col_day = (0, 0, 0) if isday else (255, 255, 255)
    col_arr = show_control(col_arr, control, environment, targets, camera, col_day)
    
    for i, target in enumerate(targets):

        col_dam = (0, 0, 255) if target.damage else (0, 0, 0)                           # Damage: red [damage], black [undamaged]
        
        col_arr = showp(col_arr, target.sW)                                             # ADD: Center
        if not setting == _3d: col_arr = show2d(col_arr, target.bbox2d, col_dam)        # ADD: 2D BB
        if not setting == _2d: col_arr = show3d(col_arr, target.bbox3d, col_dam)        # ADD: 3D BB
        if display_features: col_arr = show_features(col_arr, target, i, col_day)       # ADD: Features
    
    if save_image:        
        ann_path = basepath + "_annotated.jpeg"
        cv2.imwrite(ann_path, col_arr)
        print(ann_path)

    else:
        cv2.imshow('detect', col_arr)
        cv2.waitKey(0)    

def showp(col_arr, p):
    return cv2.circle(col_arr, tuple(p), 2, (255, 255, 255), -1)

def show2d(col_arr, bb2d, col_dam):
    thickness = 1
    xmin, ymin, xmax, ymax = bb2d
    return cv2.rectangle(col_arr, (xmin, ymin), (xmax, ymax), col_dam, thickness)

def show3d(col_arr, bb3d, col_dam):
    thickness = 1

    if len(bb3d) == 8:

        points = [# Panel: FR 
                  bb3d["ftl"], 
                  bb3d["ftr"],
                  bb3d["fbr"],
                  bb3d["ftl"],
                  bb3d["fbl"],
                  bb3d["ftr"],
                  bb3d["btr"],
                  bb3d["bbr"],
                  bb3d["fbr"],
                  bb3d["fbl"],
                  # Panel: BL
                  bb3d["bbl"], 
                  bb3d["btr"],
                  bb3d["btl"],
                  bb3d["bbr"],
                  bb3d["bbl"],
                  bb3d["btl"],
                  bb3d["ftl"]]
    else:
        points = list(bb3d.values())

    contours = np.array(points).reshape((-1, 1, 2))

    return cv2.drawContours(col_arr, [contours], 0, col_dam, thickness)

def show_features(col_arr, target, i, col_day):
    
    sW = tuple(np.array(target.sW) + np.array([0, -10]))
    
    bb3d = target.bbox3d
    cnr_key = 'ftr' if 'ftr' in bb3d.keys() else random.choice(list(bb3d.keys()))
    cnr_val = tuple(np.array(bb3d[cnr_key]) + np.array([1, 1]))

    # Features

    rotx, roty, rotz = np.degrees(target.rot).astype(int)
    text = '{:s} | Occ: {:.2f} | Rot: <{:d},{:d},{:d}> | Pix: {:d}'.format(cnr_key.capitalize(), 
                                                                           target.occluded,                                                         
                                                                           rotx,
                                                                           roty,
                                                                           rotz,
                                                                           target.pixelarea)

    col_arr = cv2.putText(col_arr, text, cnr_val, 
                          fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                          fontScale=0.5, 
                          color=col_day, 
                          thickness=1)

    col_arr = cv2.putText(col_arr, '{}'.format(i), sW, 
                          fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                          fontScale=0.7, 
                          color=(255, 255, 255), 
                          thickness=1)
    return col_arr

def show_control(col_arr, control, environment, targets, camera, col_day):

    nvehicles = len(targets)
    ndamaged = len([target for target in targets if target.damage])

    ctrl_text = "Control: ID {} | Vehicles [Damaged] {} [{}]".format(control["ids"]["test"], 
                                                                     nvehicles, 
                                                                     ndamaged) 
                                                    
    col_arr = cv2.putText(col_arr, ctrl_text, (1, 20), 
                          fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                          fontScale=0.5, 
                          color=col_day, 
                          thickness=1)

    env_text = "Environment: {} | {} [{}/{}]".format(environment["gametime"], 
                                                     environment["weather"], 
                                                     int(environment["rainlevel"]), 
                                                     int(environment["snowlevel"]))

    col_arr = cv2.putText(col_arr, env_text, (1, 40), 
                          fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                          fontScale=0.5, 
                          color=col_day, 
                          thickness=1)

    cam_rotx, cam_roty, cam_rotz = np.round(np.degrees(camera.rot)).astype(int)
    cam_text = "Camera: Altitude {:s} | Pitch {:d}".format(control["assess_factors"]["altitude"], 
                                                           cam_rotx) 

    col_arr = cv2.putText(col_arr, cam_text, (1, 60), 
                          fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                          fontScale=0.5, 
                          color=col_day, 
                          thickness=1)

    return col_arr


    






