# Description: Inspection of visual correspondence between grid and vehicle scales

import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('input_path', type=str, help='path to input image')
parser.add_argument('im_size', type=int, help='image target dimensions')
parser.add_argument('grid_size', type=int, help='anchor grid size')
parser.add_argument('-o', '--output_path', type=str, default = '', help='output path of grid-lined input image')

args = parser.parse_args()

# A. Read in and decode image

img = tf.io.read_file(args.input_path)
img = tf.io.decode_jpeg(img, channels = 3)
img = tf.image.resize(img, (args.im_size, args.im_size))

# B. Format grid

dx = dy = args.im_size // args.grid_size
grid_color = [0, 0, 0]

arr = img.numpy().astype(int)
arr[:,::dy,:] = grid_color
arr[::dx,:,:] = grid_color

# C. Display and save image

if args.output_path != '': plt.imsave(args.output_path, arr)

plt.imshow(arr)
plt.show()
