""" 
    Description: Move filtered-out output files to an 'error' subfolder
                 Move valid tfrecords to a 'tfrecords' folder
                 Perform and re-perform training-validation split

    Sample cmd instruction: python filter_to_tfrecord.py ./data/filter/ 0.8
"""

import os
import argparse

from absl import app
from pathlib import Path
from itertools import chain

from methods.data import set_split, move_test_files
from methods.image import annotate
from methods.files import extract_name, get_basepath_diff, move_files

parser = argparse.ArgumentParser()
parser.add_argument('filter_dir', type=str, help='filter directory')
parser.add_argument('train_per', type=float, default=0.8, help='training/validation split')

def main(_argv):

    args = parser.parse_args()

    parent_dir = os.path.dirname(os.path.normpath(args.filter_dir))
    output_dir = os.path.join(parent_dir, 'output')
    
    # Move corresponding filtered-out output files to error subfolder 

    filter_error_dir = os.path.join(args.filter_dir, 'error')
    output_error_dir = os.path.join(output_dir, 'error')

    if os.path.isdir(output_error_dir):

        basepaths = get_basepath_diff(filter_error_dir, 
                                      output_error_dir, 
                                      inext='_annotated.jpeg', 
                                      outext='.txt')
    else:

        basepaths = [extract_name(d, isbasepath=True) for d in 
                     Path(filter_error_dir).glob('*_annotated.jpeg')]
    
    print('new error basepaths: {}'.format(basepaths))

    if len(basepaths) > 0:

        out_names = [extract_name(d) for d in basepaths]
        out_basepaths = [os.path.join(output_dir, d) for d in out_names] 
        out_files = list(chain.from_iterable([[d + '.txt', d + '.tfrecord'] 
                                              for d in out_basepaths]))

        print('out_files: ', out_files) # ADJUST

        move_test_files(out_files, 'error', levelup=0)

    # Move valid tfrecords to a tfrecords folder

    tf_files = [str(d) for d in Path(output_dir).glob('*.tfrecord')]
    print('new tfrecords added: {}'.format(tf_files))

    if len(tf_files) > 0:

        move_test_files(tf_files, 'tfrecords', levelup=1)

        # Perform training-validation split

        tfrecord_dir = os.path.join(parent_dir, 'tfrecords')
        train_dir = os.path.join(tfrecord_dir, 'train')

        if os.path.isdir(train_dir):

            files = list(Path(tfrecord_dir).glob('*/*.tfrecord'))
            move_files(files, tfrecord_dir, iscopy=False)

        files = list(Path(tfrecord_dir).glob('*.tfrecord'))
        set_split(files, args.train_per)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass        



