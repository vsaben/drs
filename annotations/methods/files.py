"""
    Description: Move, copy or delete files | id range
"""

from pathlib import Path
import shutil
import os

def extract_name(path, isbasepath = False):

    """Extracts test name from a full file path"""

    path = str(path)
    name = os.path.basename(path)[:11]
    
    if not isbasepath:
        return name

    dir = os.path.dirname(path)
    basepath = os.path.join(dir, name)
    return basepath

def get_name(id):

    """Get test name (XXXXXXW00RD) from integer id"""

    id_str = (6 - len(str(id)))*'0' + str(id)
    return "{:s}W00RD".format(id_str)

def get_name_range(id_min, id_max):

    """Get padded string test names | id range 
    (upper bound is inclusive)"""

    names = []
    for id in range(id_min, id_max + 1):
        name = get_name(id)
        names.append(name)
    return names

def get_basepath_diff(indir, outdir, inext, outext):

    """Gets files in the input directory 
    (matching the specified regex) that do not 
    appear in the target directory"""

    in_files = Path(indir).glob('*' + inext)
    out_files = Path(outdir).glob('*' + outext)

    in_names = set([extract_name(d) for d in in_files])
    out_names = set([extract_name(d) for d in out_files])

    diff_names = in_names - out_names
    
    basepaths = [os.path.join(indir, name) for name in diff_names] 
    return basepaths


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


    

