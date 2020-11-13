"""
    Description: Prepare and visualise image annotations
    Function:    
    - PART A: Read in image and annotation files
    - PART B: Extract camera and environment properties; target instances
    - PART C: Visualise target instances and create annotation files
"""

from data import read_data
from utils import Camera
from target import generate_targets
from visualise import draw_annotations
from annotations import write_txt, write_tfrecord

def annotate(basepath, save_image = False, save_tfrecord = False, save_txt = False):

    """Prepare, visualise and save annotations

    :param basepath: file path substring common to all simulation components (eg. "*\015816W00RD")
    :option save_image: save annotated image
    :option save_tfrecord: save tfrecord
    :option save_txt: save txt

    :result: visualised, saved '*._annotated.jpeg', '.txt' and/or '.tfrecord' files 
    """

    # PART A: Read in image and annotation files =================================================

    isvalid, data, control, col_img, col_bytes, ste_arr = read_data(basepath)      
    if not isvalid: 
        print('invalid instance: ', basepath)
        return

    # PART B: Extract camera and environment properties; target instances

    camera = Camera(data["camerast"])                            
    targets = generate_targets(camera, data["targetst"], ste_arr) 
    environment = data["environmentst"]

    # PART C: Visualise target instances and create annotation files =============================
     
    draw_annotations(basepath, 
                     col_img, 
                     control, 
                     environment, 
                     targets,
                     camera, 
                     setting=2, 
                     save_image = save_image)
    
    if save_tfrecord: write_tfrecord(basepath, col_bytes, environment, camera, targets) 
    if save_txt: write_txt(basepath, control, environment, camera, targets)



