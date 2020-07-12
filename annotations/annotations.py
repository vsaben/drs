# Description: Write annotations to file
# Files:
#   A. ".txt"
#   B. ".tfrecord" 

# A: Write to txt file ------------------------------------------------------------ 

def write_txt(basepath, control, environment, targets):
    
    # > i: Control
    
    testid = control["ids"]["test"]
    nvehicles = len(targets)
    ndamaged = len([target for target in targets if target.damage])

    ctrl_txt = "Control: {:d} {:d} {:d}".format(testid, nvehicles, ndamaged)

    # > ii: Environment

    time = environment["gametime"]
    weather = environment["weather"]
    rain = environment["rain"]
    snow = environment["snow"]

    env_txt = "Environment: {:s} {:s} {:d} {:d}".format(time, weather, rain, snow)

    # > iii Camera

    cam_txt = "Camera {} {} {} "
    r_txt = "R "






    texts = [ctrl_txt, env_txt]




    # > iii: Targets

    file = basepath + ".txt"

    with open(file, "w") as f:
        f.write()


# B: Write to tfrecord -------------------------------------------------------------

import tensorflow as tf
import numpy as np

# > i: Type compatibility with tf.Example

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

def serialize_example(image, camera, targets):

    pos_x, pos_y, pos_z = extract_prop_list(targets, "pos", True)
    dim_x, dim_y, dim_z = extract_prop_list(targets, "dims", True)
    rot_x, rot_y, rot_z = extract_prop_list(targets, "rot", True)
    
    tru = extract_prop_list(targets, "truncation", False)
    occ = extract_prop_list(targets, "occluded", False)
    dam = extract_prop_list(targets, "damage", False)
    cls = extract_prop_list(targets, "vehicleclass", False)

    flat_R = list(np.hstack(camera.R))

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
               'cam_yaw': _float_feature(camera.yaw, False),
               'cam_R': _float_feature(flat_R, True),
               'cam_w': _int64_feature(camera.w, False),
               'cam_h': _int64_feature(camera.h, False),
               'cam_fx': _float_feature(camera.fx, False),
               'cam_fy': _float_feature(camera.fy, False)}

    example_proto = tf.train.Example(features = tf.train.Features(feature = feature))
    return example_proto.SerializeToString()

def extract_prop_list(classes, prop, isvec = False):
    res_list = [getattr(cls, prop) for cls in classes]
    if not isvec: return res_list
    
    res_chunks = np.vstack(res_list)
    res_x, res_y, res_z = res_chunks.T  
    return list(res_x), list(res_y), list(res_z)
    
#def tf_serialize_example(image, camera, targets):    
#    tf_string = tf.py_func(serialize_example, (image, camera, targets), tf.string)      
#    res_tf_string = tf.reshape(tf_string, ())
#    return res_tf_string

#def serialize_dataset(images, targets): 
#
#    features_dataset = tf.data.Dataset.from_tensor_slices((images, targets))
#    serialized_features_dataset = features_dataset.map(tf_serialize_example)
#
#    return serialized_features_dataset

def write_tfrecord(basepath, image, camera, targets):

    tfrecord_path = basepath + ".tfrecord"

    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        example = serialize_example(image, camera, targets) 
        writer.write(example)

    print(tfrecord_path)

#def convert_to_tfrecord(input_directories, output_dir, set):
#
#    ps = ProcessSet(input_directories)
#    
#    images = [] 
#    n = len(ps.image_paths)

#    for i, d in enumerate(ps.image_paths):
#        images += [process_image_paths(d)]
#        progress = (i + 1) / n
#        print("{:.3f}".format(progress))
    
#    labels = ps.labels

#    image_batches = DivideListIntoChunks(images, 200)
#    label_batches = DivideListIntoChunks(labels, 200)

#    i = 1
#    for image_batch, label_batch in zip(image_batches, label_batches):    
#        
#        no_images = len(list(label_batch))
#        batch_path = str(output_dir / (set + "_{:d}_{:d}.tfrecord".format(i, no_images)))
#        i += 1

#        write_tfrecord_to_file(batch_path, image_batch, label_batch)   













    


















