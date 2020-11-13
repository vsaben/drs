""" 
    Description: Prepare and visualise image annotations; 
                 Move/sort file components  

    Function: CMD line integration / argparser components 
    - PART A: Shared 
    - PART B: Move/sort
    - PART C: Annotations
    - PART D: Main
"""

import os
import argparse

from image import annotate
from files import get_names, change_files

# PART A: Shared =======================================================================================================

parser = argparse.ArgumentParser()
parser.add_argument('input_dir', type=str)
parser.add_argument("id_min", type=int)
parser.add_argument("id_max", type=int)

subparsers = parser.add_subparsers(help="move, copy or annotate files", dest="cmd")

# PART B: Move/sort ====================================================================================================

move_parser = subparsers.add_parser('F', help = "move files between directories matching a specific regex")

group = move_parser.add_mutually_exclusive_group()
group.add_argument("-c", "--copy", action="store_true", help="copy files")
group.add_argument("-m", "--move", action="store_true", help="move files")
group.add_argument("-d", "--delete", action="store_true", help="delete files")

move_parser.add_argument("-o", "--output_dir", type=str, default=False, required=False) # Not needed for deletions

move_parser.add_argument("-a", "--annotated", action="store_const", const=0, default=False, help="annotated colour images")
move_parser.add_argument("-i", "--image", action="store_const", const=1, default=False, help="unannotated colour images")
move_parser.add_argument("-tf", "--tfrecord", action="store_const", const=2, default=False, help="tfrecords")
move_parser.add_argument("-tx", "--txt", action="store_const", const=3, default=False, help="txts")
move_parser.add_argument("-js", "--json", action="store_const", const=4, default=False, help="json")
move_parser.add_argument("--depth", action="store_const", const=5, default=False, help="depth image")
move_parser.add_argument("--stencil", action="store_const", const=6, default=False, help="stencil image")
move_parser.add_argument("--tif", action="store_const", const=7, default=False, help="unannotated colour tif image")
move_parser.add_argument("--all", action="store_const", const=8, default=False, help="all existing related test files")

# PART C: Annotations ===================================================================================================

ann_parser = subparsers.add_parser("A", help = "Annotate files in directory")
ann_parser.add_argument("-o", "--output_dir", type=str, default=False, required=False) # If specified, change output directory of processed files
ann_parser.add_argument("-a", "--annotated", action="store_true", help="save annotated image")
ann_parser.add_argument("-tf", "--tfrecord", action="store_true", help="save tfrecord")
ann_parser.add_argument("-tx", "--txt", action="store_true", help="save txt file")

# PART D: Main ==========================================================================================================

args = parser.parse_args()

names = get_names(args.id_min, args.id_max)
basepaths = [os.path.join(args.input_dir, d) for d in names]

if args.cmd == 'F': change_files(args, basepaths)    
if args.cmd == 'A':                
    for d in basepaths:
        annotate(d, save_image = args.annotated, 
                    save_tfrecord = args.tfrecord, 
                    save_txt = args.txt)


