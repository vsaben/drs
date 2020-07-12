# Description: Image Control
# Function:    
#    A. Read in image and annotation files
#    B. Fomulate camera properties
#    C. Extract target instances
#    D. Visualise target instances
#    E. Create annotation file

PKG_DIR = "C:\\Users\\Vaughn\\projects\\work\\drs\\annotations"

import sys
sys.path += [PKG_DIR]

from data import read_data, Camera
from target import GenerateTargets
from visualise import ShowBB
from annotations import write_tfrecord

def Annotate(basepath, save_image = False, save_tfrecord = False):

    data, col_img, col_bytes, ste_arr = read_data(basepath)      # A: Read in image and annotation files
    camera = Camera(data["camerast"])                            # B: Formulate camera properties
    targets = GenerateTargets(camera, data["targetst"], ste_arr) # C: Extract target instances
    
    # D: Visualise target instances

    if len(targets) > 0:
        control = data["controlst"]
        environment = data["environmentst"]
    
        ShowBB(basepath, col_img, control, environment, targets, setting=2, save_image = save_image)
    
    if save_tfrecord:                                            # E: Write annotations to tfrecord           
        write_tfrecord(basepath, col_bytes, camera, targets) 




