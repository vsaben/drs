"""
    Description: Visualise model output
"""

import tensorflow as tf
import numpy as np
import cv2

from methods._data.reformat import compute_bb3d, Q2E
from absl import logging

DIMS = ["ftl", "ftr", "fbl", "fbr",   
        "btl", "btr", "bbl", "bbr"]

def get_ann_images(padded_anns, cameras, images, cfg, size = None):

    """Get annotated images given padded annotations, images and 
    camera information.

    :param gt_padded_anns: [rpn (input = 5, output = 6), head (10)]
    :param cameras: camera dictionaries (separate)
    :param images: batched tf images 
   
    :result: batched annotated images
    """
    
    ncol = padded_anns.shape[2]
    mode = 'input' if ncol == 15 else 'output' 

    if mode == 'input':
        col_udam = (105, 105, 105)
        col_dam = (203, 204, 255)
    elif mode == 'output':
        col_udam = (0, 0, 0)
        col_dam = (0, 0, 255)

    arr_list = []
    cameras = unbatch_dict(cameras)

    for c, camera in enumerate(cameras):
        
        img = images[c]
        padded_ann = padded_anns[c]

        arr = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR)        

        camera = cameras[c]
        if size is not None:
            camera['w'] = size
            camera['h'] = size

        # Unpad

        gt_cond = padded_ann[:, 2]
        pos_ix = tf.where(gt_cond > 0)       
        ann = tf.gather_nd(padded_ann, pos_ix)

        # Annotations
        
        nann = tf.shape(ann)[0].numpy()
        ndamaged =  np.sum(ann[:, -10], dtype = np.int32) if nann > 0 else 0
        arr = show_control(arr, nann, ndamaged, camera, mode, col_udam)       

        logging.info('image: {:d} | detections (damaged): {:d} ({:d})'.format(c, nann, ndamaged))  

        if nann > 0:

            rpn = ann[..., :4] 
            pose = ann[..., -10:]

            bb2d = norm_to_screen(rpn, camera)                        
            score = ann[..., 4] if mode == 'output' else [None]*nann                
            dam = ann[..., -11]
            
            cpos = pose_to_cpos(pose, camera, cfg)       
            bb3d = tf.map_fn(lambda e: compute_bb3d(camera, e), cpos, tf.int16)

            center = pose[...,:2]        
            sw = center * np.array([camera['w'] - 1, camera['h'] - 1]) 

            # Draw
       
            for i in range(nann):

                col = col_dam if dam[i] else col_udam

                arr = showp(arr, sw[i], col)                               # center
                arr = show2d(arr, bb2d[i], col)                            # bb2d

                b3 = tf.stack(bb3d[i], axis = 0).numpy()
                b3 = dict(zip(DIMS, b3))
                arr = show3d(arr, b3, col)                                 # bb3d

                b3_strip = {k: v for k, v in b3.items() if is_on_screen(v, camera)}                                                   
                arr = show_features(arr, i, 
                                    sw[i], b3_strip, cpos[i], 
                                    col, score[i])                         # features


                update_rpn = '{:s}: {:d} bbox: {} dam: {:d}'.format(mode, i, bb2d[i], int(dam[i])) + (
                    ' score: {:.2f}'.format(score[i]) if mode == 'output' else '')

                update_pose = ' pos: {} dim: {} rot: {}'.format(
                    lst2floatstr(cpos[i][0:3], 2),
                    lst2floatstr(cpos[i][3:6], 2),
                    lst2floatstr(np.degrees(cpos[i][6:]), 2)) 

                logging.info(update_rpn + update_pose)

        arr_list.append(arr)         
            
    return tf.stack(arr_list, axis=0)

def unbatch_dict(batched_dict, ilen = 0):

    """Unbatches amalgamated dictionaries"""
  
    nbatch = list(batched_dict.values())[ilen].shape[0]

    return [{k: v[i] for k, v in batched_dict.items()} 
            for i in range(nbatch)]
  
def lst2floatstr(lst, decimal):

    """Converts iterator/list components to rounded floats
    (for display)"""

    rstr = ":.{:d}f".format(decimal)
    return [float(("{" + rstr + "}").format(n)) for n in lst]


def norm_to_screen(rpn_roi, camera):

    """Scale normalised rpn roi to screen
    coordinates. 
    
    [0, 1] --> [W - 1, H - 1]

    :param rpn_roi: [nbatch, MAX_GT_INSTANCES, 4]
    :param camera: camera property dictionary

    :result: screen-scaled 2D bbox coordinates
    """

    W_1 = camera['w'] - 1
    H_1 = camera['h'] - 1

    factor = np.array([W_1, H_1, W_1, H_1])
    return np.round(rpn_roi * factor).astype(int)

def S2C(pose, camera):

    """Screen to camera coordinates

    :param center: normalised center coordinates
    :param Z: predicted depth values
    :param cam: [W (1) + H (1) + fx (1) + fy (1) = 4]

    :result: [X, Y, Z] in camera coordinates 
    """

    center = pose[:, :2] 
    Z = tf.expand_dims(pose[:, 2], axis=1)

    W = camera['w']
    H = camera['h']
    fx = camera['fx']
    fy = camera['fy'] 

    x, y = tf.split(center * tf.constant([W - 1, H - 1], tf.float32), 2, axis = -1)

    X = ((2*x)/(W - 1) - 1)*(Z/fx)
    Y = (1 - (2*y)/(H - 1))*(Z/fy)

    return tf.concat([X, Y, Z], axis = -1)

def pose_to_cpos(pose, camera, cfg):

    """Converts predicted pose output to an annotation format

    :param pose: [detect, 10] 
         <center (2) {normalised to image} + 
          depth (1) + 
          dims (3) {absolute/relative} + 
          quat (4) = 10>
    :param camera: camera property dictionary

    :result: pose annotations in their original format
             {ndetect, pos (3) + dims (3) + rot (3)}
    """

    # 1-3: Position

    pos = S2C(pose, camera)

    # 4-6: Dimensions 

    dim = pose[..., 3:6]
    if cfg['USE_DIM_ANCHOR']:
        med_dims = tf.constant(cfg['DIM_ANCHOR'], tf.float32)
        dim += med_dims    

    # 7-9: Rotation

    quat = pose[..., -4:]
    rot = Q2E(quat)

    return tf.concat([pos, dim, rot], axis=-1)

def is_on_screen(point, camera):

    W = camera['w']
    H = camera['h']

    x, y = point
    
    boolx = (0 <= x <= W - 1)
    booly = (0 <= y <= H - 1)

    return boolx and booly

# Drawing tools ==================================================================

def showp(arr, p, pixcol):

    """Draw point on array"""

    return cv2.circle(arr, tuple(p), 2, pixcol, -1)

def show2d(arr, bb2d, col):

    """Draw 2D bounding box"""

    xmin, ymin, xmax, ymax = bb2d
    return cv2.rectangle(arr, (xmin, ymin), (xmax, ymax), col, thickness = 1)

def show3d(arr, bb3d, col):

    """Draw 3D bounding box with contours"""    

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

    contours = np.array(points).reshape((-1, 1, 2)).astype(int)
   
    return cv2.drawContours(arr, [contours], 0, col, thickness = 1)

def show_features(arr, i, sw, bb3d, cpos_elem, col, score = None):
    
    sw = np.array(sw) + np.array([0, -5])
    sw = tuple(np.round(sw).astype(int))

    cnr_key = 'ftr' if 'ftr' in bb3d.keys() else random.choice(list(bb3d.keys()))
    cnr_val = tuple(np.array(bb3d[cnr_key]) + np.array([1, 1]))

    # Features

    rotx, roty, rotz = np.degrees(cpos_elem[6:]).astype(int)
    posx, posy, posz = cpos_elem[:3]

    text = '{:s} | Pos: <{:.1f},{:.1f},{:.1f}> | Rot: <{:d},{:d},{:d}>'.format(
        cnr_key.capitalize(), 
        posx,
        posy,
        posz,
        rotx,
        roty,
        rotz)

    if score is not None:
        text += ' | Score: {:.2f}'.format(score)

    arr = cv2.putText(arr, text, cnr_val, 
                      fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                      fontScale=0.75, 
                      color=col, 
                      thickness=1)

    arr = cv2.putText(arr, '{:d}'.format(i), sw, 
                      fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                      fontScale=0.75, 
                      color=col, 
                      thickness=1)
    return arr

def show_control(arr, nvehicles, ndamaged, camera, mode, col):

    ystart = 0 if mode == 'input' else 20

    # Control

    ctrl_text = "Vehicles [Damaged] {} [{}]".format(nvehicles, ndamaged)                                                     
    arr = cv2.putText(arr, ctrl_text, (1, ystart + 10), 
                      fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                      fontScale=0.75, 
                      color=col, 
                      thickness=1)

    # Camera

    cam_rotx, cam_roty, cam_rotz = np.round(np.degrees(camera['rot'])).astype(int)
    cam_text = "Camera Pitch: {:d}".format(cam_rotx) 

    arr = cv2.putText(arr, cam_text, (1, ystart + 20), 
                      fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                      fontScale=0.75, 
                      color=col, 
                      thickness=1)
    return arr