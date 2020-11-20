""" 
    Description: Move/sort file components 
"""

import os
import argparse

from methods.files import get_names, get_files, move_files, del_files

parser = argparse.ArgumentParser()
parser.add_argument('input_dir', type=str)
parser.add_argument("id_min", type=int)
parser.add_argument("id_max", type=int)

group = parser.add_mutually_exclusive_group()
group.add_argument("-c", "--copy", action="store_true", help="copy files")
group.add_argument("-m", "--move", action="store_true", help="move files")
group.add_argument("-d", "--delete", action="store_true", help="delete files")

parser.add_argument("-o", "--output_dir", type=str, default=False, required=False) # Not needed for deletions

parser.add_argument("-a", "--annotated", action="store_const", const=0, default=False, help="annotated colour images")
parser.add_argument("-i", "--image", action="store_const", const=1, default=False, help="unannotated colour images")
parser.add_argument("-tf", "--tfrecord", action="store_const", const=2, default=False, help="tfrecords")
parser.add_argument("-tx", "--txt", action="store_const", const=3, default=False, help="txts")
parser.add_argument("-js", "--json", action="store_const", const=4, default=False, help="json")
parser.add_argument("--depth", action="store_const", const=5, default=False, help="depth image")
parser.add_argument("--stencil", action="store_const", const=6, default=False, help="stencil image")
parser.add_argument("--tif", action="store_const", const=7, default=False, help="unannotated colour tif image")
parser.add_argument("--all", action="store_const", const=8, default=False, help="all existing related test files")

args = parser.parse_args()

names = get_names(args.id_min, args.id_max)
basepaths = [os.path.join(args.input_dir, d) for d in names]

if (args.move or args.copy) and not args.output_dir:  
    
    raise Exception("No output directory provided in move/copy operation")
    
# Determine file type indices

if not args.all:
    file_options = [args.annotated, args.image, args.tfrecord, args.txt, 
                    args.json, args.depth, args.stencil, args.tif]
    file_indices = [d for d in file_options if type(d) == int]
else:
    file_indices = range(8)

# Perform operation on files

files = get_files(basepaths, file_indices)
    
if args.move: 
    move_files(files, args.output_dir, iscopy = False)
if args.copy: 
    move_files(files, args.output_dir, iscopy = True)
if args.delete: 
    del_files(files)
 


        



