# Description: Horizontal flip augmentation

import tensorflow as tf

@tf.function
def horizontal_init(cpos, x, y1, camera, features, environment):

    # Aug [-pos_x, pos_y, pos_z, dim_x, dim_y, dim_z, rot_x, -rot_y, -rot_z] 

    # Function: Performs a left-right flip of the image and associated annotations

    x = tf.image.flip_left_right(x)
    
    if tf.equal(tf.size(cpos), 0): 
        return cpos, x, y1, camera, features, environment 

    # Original: [xmin, ymin, xmax, ymax, damage]
    # New: [1 - xmax, ymin, 1 - xmin, ymax, damage] 

    y1 = tf.stack([1 - y1[:, 2],              # xmin
                   y1[:, 1],                  # ymin
                   1 - y1[:, 0],              # xmax
                   y1[:, 3],                  # ymax
                   y1[:, 4]], axis = 1)       # damage
    
    # Original: [posx*, posy, posz, dimx, dimy, dimz, rotx, roty*, rotz*]
    # Change: Items denoted with *
    # > posx --> -posx (as relative to camera)
    # > roty --> -roty (roll reflected across y-axis. Note it is symmetrically centered.)
    # > rotz --> -rotz (yaw reflected. Note it is symmetrically centered.)
    
    cpos = tf.stack([-cpos[:, 0],             # posx
                      cpos[:, 1],             # posy
                      cpos[:, 2],             # posz 
                      cpos[:, 3],             # dimx
                      cpos[:, 4],             # dimy 
                      cpos[:, 5],             # dimz
                      cpos[:, 6],             # rotx
                     -cpos[:, 7],             # roty
                     -cpos[:, 8]], axis = 1)  # rotz  
         
    return cpos, x, y1, camera, features, environment

def horizontal_flip(x, cpos, features, camera, environment):
    
    x = tf.image.flip_left_right(x)

    cpos = tf.stack([-cpos[:, 0],             # posx
                      cpos[:, 1],             # posy
                      cpos[:, 2],             # posz 
                      cpos[:, 3],             # dimx
                      cpos[:, 4],             # dimy 
                      cpos[:, 5],             # dimz
                      cpos[:, 6],             # rotx
                     -cpos[:, 7],             # roty
                     -cpos[:, 8]], axis = 1)  # rotz

    return x, cpos, features, camera, environment


