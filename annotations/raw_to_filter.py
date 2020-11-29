""" 
    Description: Prepare and visualise image annotations

    Sample cmd instructions: 

        single instance: python raw_to_filter.py ./data/raw/ 1 -o ./data/examples/ -a -tx
              
        raw_to_filter: python raw_to_filter.py ./data/raw/ U  
            updates all filter, output and tfrecord files
            and sorts them by:
                raw: correct + error 
                filter: passed raw check (to be manually filtered)
            for any new data found in raw
    """

import os
import argparse

from pathlib import Path
from itertools import chain

from methods.data import move_test_files
from methods.image import annotate
from methods.files import get_name, extract_name, get_basepath_diff

parser = argparse.ArgumentParser()
parser.add_argument('rawdir', type=str, help='raw directory')

subparsers = parser.add_subparsers(help="annotate a single image or perform raw_to_filter directory update", dest="cmd")

# Single instance

single_parser = subparsers.add_parser('S', help = "generates annotation files for a single test instance")
single_parser.add_argument("id", type=int)

single_parser.add_argument("-a", "--annotated", action="store_true", help="save annotated image")
single_parser.add_argument("-tf", "--tfrecord", action="store_true", help="save tfrecord")
single_parser.add_argument("-tx", "--txt", action="store_true", help="save txt file")

# Directory update

update_parser = subparsers.add_parser('U', help = 'perform raw_to_filter directory update')

def main():

    args = parser.parse_args()

    if args.cmd == 'S':

        name = get_name(args.id)
        basepath = os.path.join(args.rawdir, name)

        test_file = basepath + '.json'

        if not os.path.isfile(test_file):
            print("some/all raw data for test {:d} not found")
            return

        status = annotate(basepath, save_image = args.annotated, 
                                    save_tfrecord = args.tfrecord, 
                                    save_txt = args.txt)

    elif args.cmd == 'U':

        parent_dir = os.path.dirname(os.path.normpath(args.rawdir))
        output_dir = os.path.join(parent_dir, 'filter')

        if os.path.isdir(output_dir):

            basepaths = get_basepath_diff(args.rawdir, 
                                          args.rawdir, 
                                          inext='.json', 
                                          outext='_stencil.jpeg')

        else:

            basepaths = [extract_name(d, isbasepath=True) for d in 
                         Path(args.rawdir).glob('*.json')]

        print("new basepaths: {}".format(basepaths))

        for d in basepaths:
            status = annotate(d, save_image = True, 
                                 save_tfrecord = True, 
                                 save_txt = True)

        # Move annotated and colour files for manual filtering

        filter_files = list(chain.from_iterable([[d + '_colour.jpeg', 
                                                  d + '_annotated.jpeg'] 
                                                 for d in basepaths]))

        move_test_files(filter_files, 'filter', levelup = 1)
       
        # Move tfrecords and txt files to output

        out_files = list(chain.from_iterable([[d + '.tfrecord', 
                                               d + '.txt'] 
                                              for d in basepaths]))

        move_test_files(out_files, 'output', levelup = 1)


main()


