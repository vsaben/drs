"""
    Description: Move, copy or delete files | id range
"""

from pathlib import Path
import shutil
import os

def get_names(id_min, id_max):

    """Get padded string test names | id range 
    (upper bound is inclusive)"""

    names = []
    for id in range(id_min, id_max + 1):
        id_str = (6 - len(str(id)))*'0' + str(id)
        names.append("{:s}W00RD".format(id_str))
    return names

FILE_TYPES = ["_annotated.jpeg", 
              "_colour.jpeg", 
              ".tfrecord", 
              ".txt", 
              ".json", 
              "_depth.tif", 
              "_stencil.tif", 
              "_colour.tif"]

def get_files(basepaths, file_indices):

    """Get all files associated with basepaths | file type indices"""

    exts = [FILE_TYPES[ind] for ind in file_indices]    
    files = []
    for basepath in basepaths:        
        fullpaths = [basepath + ext for ext in exts]
        fullexist = [d for d in fullpaths if os.path.isfile(d)]
        files += fullexist
    return files

def move_files(files, output_dir, iscopy=True):

    """Move or copy files to the output directory"""

    op = shutil.copy if iscopy else shutil.move

    for file in files:
        name = file[(file.rfind("\\") + 1):]       
        new_path = os.path.join(output_dir, name)
        op(file, new_path)
        print(new_path)

def del_files(files):

    """Delete files in file list"""

    for file in files:
        os.remove(file)
        print("Deleted: ", file)


def change_files(args, basepaths):

    """"""

    # Check output directory provided (if necessary)

    if (args.move or args.copy) and not args.output_dir:  
        print("No output directory provided in move/copy operation")
        return

    # Determine file type indices

    if not args.all:
        file_options = [args.annotated, args.image, args.tfrecord, args.txt, 
                        args.json, args.depth, args.stencil, args.tif]
        file_indices = [d for d in file_options if type(d) == int]
    else:
        file_indices = range(8)

    # Perform operation on files

    files = get_files(basepaths, file_indices)
    
    if args.move: move_files(files, args.output_dir, iscopy = False)
    if args.copy: move_files(files, args.output_dir, iscopy = True)
    if args.delete: del_files(files)

