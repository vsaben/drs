"""
    Description: Write annotations to file
    Function: To
    - PART A: ".txt" (readable check)
    - PART B: ".tfrecord" (model training and evaluation)
"""

import tensorflow as tf
import numpy as np

# PART A: Write to txt file ================================================================

def write_txt(basepath, control, environment, camera, targets):
    
    """Write annotations to a '.txt' file. Sections include:
        
        > Control
        > Environment
        > Camera
        > Targets
    
     """

    testid = control["ids"]["test"]
    nvehicles = len(targets)
    ndamaged = len([target for target in targets if target.damage])
    ctrl_txt = "CONTROL | Test ID: {:d} | Number of Vehicles (damaged): {:d} ({:d})".format(
        testid, 
        nvehicles, 
        ndamaged)

    env_txt = "ENVIRONMENT | Game Time: {:s} | Weather (rain/snow): {:s} ({:d}/{:d})".format(
        environment["gametime"], 
        environment["weather"],
        int(environment["rainlevel"]),
        int(environment["snowlevel"]))

    cam_txt = "CAMERA | Altitude: {:s} | Rot: {:s} | Dimensions: {:d} x {:d} | VFOV: {:.0f}".format(
        control["assess_factors"]["altitude"],
        arr_to_str(np.degrees(camera.rot), 2),
        camera.w, 
        camera.h, 
        np.degrees(camera.vfov))
    r_txt = "R: {:s}".format(arr_to_str(camera.R, 2, isvec=False))


    tar_txts = ["ID: {:d} | Pos: {:s} | Dim: {:s} | Rot: {:s} | Trunc: {:.2f} | Occl: {:.2f} | Damage: {:d} | Vehicle Class: {:s}".format(
        i, 
        arr_to_str(t.pos, 2), 
        arr_to_str(t.dims, 2), 
        arr_to_str(np.degrees(t.rot), 2), 
        t.truncation, 
        t.occluded, 
        t.damage, 
        t.vehicleclass) for i, t in enumerate(targets)]

    texts = [ctrl_txt, env_txt, cam_txt, r_txt] + tar_txts
    file = basepath + ".txt"

    with open(file, "w") as f:
        for txt in texts: 
            f.write(txt + "\n")

    print(file)

def arr_to_str(arr, decimal=0, isvec=True):

    """Summarises and reformats array as printable string
    
    :param arr: numpy array
    :param decimal: number of decimals to round array elements to

    :result: <a,b,c>
    """

    arr = np.round(arr, decimal)

    if isvec:
        arr_txt = ','.join([str(elem) for elem in arr])
        res_txt = "<{:s}>".format(arr_txt)
    else:
        arr_txt = [' '.join([str(elem) for elem in vec]) for vec in arr]
        res_txt = '\n   '.join(arr_txt)
  
    return res_txt

# PART B: Write annotations to tfrecord =============================================

# i: Type compatibility with tf.Example

def _int64_feature(value, islist):
    if not islist: value = [value]
    return tf.train.Feature(int64_list = tf.train.Int64List(value = value))

def _float_feature(value, islist):
    if not islist: value = [value]
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def _bytes_feature(value, islist, isstr = False):
    if not islist: value = [value]
    if isstr: value = [str.encode(v) for v in value]
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = value))

# > ii: Serialize example

def serialize_example(image, environment, camera, targets):

    pos_x, pos_y, pos_z = extract_prop_list(targets, "pos", True)
    dim_x, dim_y, dim_z = extract_prop_list(targets, "dims", True)
    rot_x, rot_y, rot_z = extract_prop_list(targets, "rot", True)
    
    tru = extract_prop_list(targets, "truncation", False)
    occ = extract_prop_list(targets, "occluded", False)
    dam = extract_prop_list(targets, "damage", False)
    cls = extract_prop_list(targets, "vehicleclass", False)

    feature = {'image': _bytes_feature(image, False),
               
               'pos_x': _float_feature(pos_x, True),
               'pos_y': _float_feature(pos_y, True),
               'pos_z': _float_feature(pos_z, True),
               'dim_x': _float_feature(dim_x, True),
               'dim_y': _float_feature(dim_y, True),
               'dim_z': _float_feature(dim_z, True),
               'rot_x': _float_feature(rot_x, True),
               'rot_y': _float_feature(rot_y, True),
               'rot_z': _float_feature(rot_z, True),
               
               'tru': _float_feature(tru, True),
               'occ': _float_feature(occ, True),
               'dam': _int64_feature(dam, True),
               'cls': _bytes_feature(cls, True, True), 
               
               'cam_rot': _bytes_feature(camera.rot.tobytes(), False),
               'cam_w': _int64_feature(camera.w, False),
               'cam_h': _int64_feature(camera.h, False),
               'cam_vfov': _float_feature(camera.vfov, False),

               'env_time': _bytes_feature(environment["gametime"], False, True), 
               'env_wthr': _bytes_feature(environment["weather"], False, True),
               'env_rain': _int64_feature(int(environment["rainlevel"]), False),
               'env_snow': _int64_feature(int(environment["snowlevel"]), False)}

    example_proto = tf.train.Example(features = tf.train.Features(feature = feature))
    return example_proto.SerializeToString()

def extract_prop_list(classes, prop, isvec = False):    
    res_list = [getattr(cls, prop) for cls in classes]
    if not isvec: return res_list
    
    if len(classes) == 0: return [], [], []
    res_chunks = np.vstack(res_list)
    res_x, res_y, res_z = res_chunks.T  
    return list(res_x), list(res_y), list(res_z)

def write_tfrecord(basepath, image, environment, camera, targets):

    tfrecord_path = basepath + ".tfrecord"

    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        example = serialize_example(image, environment, camera, targets) 
        writer.write(example)

    print(tfrecord_path)












    


















