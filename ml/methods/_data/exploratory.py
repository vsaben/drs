"""
    Description: Data exploration 
    - Determine: yolo anchors (given target image size), 
                 maximum detections 
                 class weighting
    - Number: images 
              detections (per dataset and image)
    - Median: dimension anchors
    - Distribution: detections 
                    damaged detections
                    weather
                    vehicle class
                    rain 
                    snow 
                    occlusion
                    truncation
    
    Functions:
    - PART A: Read in, extract and process data
    - PART B: Calculate kmean anchors 
    - PART C: Write exploratory results (incl. anchors) to file and 
              update configuration file if specified
    - PART D: Summary operations for train.py and cmd line use
"""

import numpy as np
import sys
import os
import random 
import argparse
import tensorflow as tf

sys.path += ["C:\\Users\\Vaughn\\projects\\work\\drs\\ml"] # Temporary

from config import setattrs
from methods.data import read_and_parse
from methods._data.reformat import get_bb2d, VEHICLE_CLASSES, WEATHER_CLASSES

# PART A: Read in and extract data ========================================================= 


def extract_data(x, cpos, features, camera, environment):
    """Extract data elements and summary information

    :input cpos: [posx, posy, posz, dimx, dimy, dimz, rotx, roty, rotz]
    :input features: [dam, occ, tru, cls]

    :result: per-image element and summary output
    :result box_wh: bbox widths and heights 
    :result N: number of detections
    :result ndam: number of damaged detections
    :result occ: vehicle occlusions (0 - 0.75)
    :result trufreq: truncation level frequencies
    :result cfreq: vehicle class frequencies
    :result dims: x, y and z dims
    """

    # Anchors

    bb2d = get_bb2d(camera, cpos)
    bb2d = tf.transpose(bb2d)    
    box_wh = bb2d[:, 2:4] - bb2d[:, :2] 
    
    # Number of images and damaged instances

    N = tf.gather(tf.shape(features), 0)
    ndam = tf.reduce_sum(features[:, 0])

    # Features

    occ = features[:, 1]   
    tru = tf.cast(features[:, 2] * 2, tf.int32)
    trufreq = tf.math.bincount(tru, minlength = 5, maxlength = 5)
        
    cls = tf.cast(features[:, 3], tf.int32)
    ncls = len(VEHICLE_CLASSES)
    cfreq = tf.math.bincount(cls, minlength = ncls, maxlength = ncls)

    # Dimension anchor

    dims = cpos[:, 3:6]
    
    # Environment
    
    wthr, rain, snow = tf.split(environment, 3)
    nwthr = len(WEATHER_CLASSES)
    wfreq = tf.math.bincount(wthr, minlength = nwthr, maxlength = nwthr)

    # Maximal depth

    depth = cpos[:, 2]

    return [box_wh, N, ndam, occ, trufreq, cfreq, dims, wfreq, rain, snow, depth] 

def process_data(data_path, image_size, nclusters):
    
    """Process data into exploratory/summary statistics

    :param data_path: data directory
    :param image_size: target resized image dimensions

    :result: result dictionary  
    :result box_wh: [w, h]
    :result nimage: number of images
    :result ndetect:
    :result median_detect:
    :result ndamaged:
    :result median_ndamaged:
    :result :
    """

    # Read in and partially process data  

    ds = read_and_parse(data_path, "train", image_size)                   # [x, cpos, features, camera, environment]
    ds_out = list(ds.map(extract_data))                                   # [box_wh, N, ndam, occ, trufreq, cfreq, dims, wfreq, rain, snow, depth]                                   
    ds_lst = list(zip(*ds_out))

    # Transform data into its desired form

    box_wh = np.vstack(ds_lst[0])
    centroids_lst = get_anchors(box_wh, nclusters)
    
    arr_n = np.transpose(ds_lst[1:3])     
    nimage = arr_n.shape[0]

    ndetect = int(np.sum(arr_n[:, 0]))
    ndamaged = int(np.sum(arr_n[:, 1]))

    tru_freq = get_freq(ds_lst, 4)
    cls_freq = get_freq(ds_lst, 5)
    wthr_freq = get_freq(ds_lst, 7) 

    arr_dims = np.vstack(ds_lst[6])
    median_dims = np.median(arr_dims, axis = 0)

    n_fivesum = get_fivesum(arr_n, 0, is_n=True)
    ndam_fivesum = get_fivesum(arr_n, 1, is_n=True)

    occ_fivesum = get_fivesum(ds_lst, 3)
    rain_fivesum = get_fivesum(ds_lst, 8)
    snow_fivesum = get_fivesum(ds_lst, 9)

    arr_rain = np.concatenate(ds_lst[8])
    median_ifrain = np.median(arr_rain[arr_rain > 0], axis = 0)

    arr_snow = np.concatenate(ds_lst[9])
    median_ifsnow = np.median(arr_snow[arr_snow > 0], axis = 0)

    max_depths = np.concatenate(ds_lst[10])
    max_depth = np.max(max_depths)

    # Result dictionary

    res_dict = {"centroids_lst": centroids_lst, 
                "nimage": nimage, 
                "ndetect": ndetect, 
                "n_fivesum": n_fivesum,
                "ndam_fivesum": ndam_fivesum,
                "ndamaged": ndamaged, 
                "tru_freq": tru_freq, 
                "cls_freq": cls_freq, 
                "wthr_freq": wthr_freq, 
                "median_dims": median_dims, 
                "occ_fivesum": occ_fivesum, 
                "rain_fivesum": rain_fivesum,
                "snow_fivesum": snow_fivesum,
                "median_ifrain": median_ifrain,
                "median_ifsnow": median_ifsnow, 
                "max_depth": max_depth}

    return res_dict

def get_freq(ds_lst, i):
    arr = np.vstack(ds_lst[i])
    return np.sum(arr, axis = 0)

def get_fivesum(ds_lst, i, is_n=False):
    arr = np.concatenate(ds_lst[i]) if not is_n else ds_lst[:, i]
    return np.percentile(arr, [0, 25, 50, 75, 100])
    

# PART B: Calculate kmean anchors  =======================================================

# Adaptation of ... [ADJUST]

def calculate_anchors(box_wh, nclusters, eps = 0.005):
    centroids = np.vstack(random.sample(list(box_wh), nclusters))
    return kmeans(box_wh, centroids, eps)

def IOU(x, centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w,c_h = centroid
        w,h = x
        if c_w>=w and c_h>=h:
            similarity = w*h/(c_w*c_h)
        elif c_w>=w and c_h<=h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w<=w and c_h>=h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape
    return np.array(similarities) 

def avg_IOU(X,centroids):
    n,d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        #note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum+= max(IOU(X[i],centroids)) 
    return sum/n
   
def kmeans(X, centroids, eps = 0.005):
    
    N = X.shape[0]
    iterations = 0
    k,dim = centroids.shape
    prev_assignments = np.ones(N)*(-1)    
    iter = 0
    old_D = np.zeros((N,k))

    while True:
        D = [] 
        iter+=1           
        for i in range(N):
            d = 1 - IOU(X[i],centroids)
            D.append(d)
        D = np.array(D) # D.shape = (N,k)
        
        print("iter {}: dists = {}".format(iter,np.sum(np.abs(old_D-D))))
            
        #assign samples to centroids 
        assignments = np.argmin(D,axis=1)
        
        if (assignments == prev_assignments).all() :
            print("Centroids = ",centroids)
            return centroids

        #calculate new centroids
        centroid_sums=np.zeros((k,dim),np.float)
        for i in range(N):
            centroid_sums[assignments[i]]+=X[i]        
        for j in range(k):            
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j))
        
        prev_assignments = assignments.copy()     
        old_D = D.copy()  

def get_anchors(box_wh, nclusters):
    
    res_anchors = []
    res_avg_iou = []
    
    for ncluster in range(1, nclusters + 1):  
        anchors = calculate_anchors(box_wh, ncluster)
        res_anchors.append(anchors)
        
        avg_iou = avg_IOU(box_wh, anchors)
        res_avg_iou.append(avg_iou)        
        
    return list(zip(range(1, nclusters + 1), res_anchors, res_avg_iou))

# PART C: Write exploratory results to file and update configuration file =====================

def write_explore(output_dir, res_dict, image_size, cfg = None):

    """Write exploratory results (including anchors) to file. Update configuration class if
    specified.

    :param output_dir: file output directory
    :param res_dict: exploratory result dictionary (see function process_data)

    :result: exploratory .txt results file and updated cfg class (if specified)
    """

    r = res_dict

    # Write to .txt

    output_path = os.path.join(output_dir, 'exploratory.txt')
    f = open(output_path, 'w')

    f.write("Number of Images: {:d}\n".format(r['nimage']))
    
    # > Detections
    
    f.write("Number of Detections (damaged): {:d} ({:d})\n".format(r['ndetect'], r['ndamaged']))
    
    max_boxes = int(r['n_fivesum'][-1])
    n_txt = " ".join(map(str, np.round(r['n_fivesum'], 2).tolist()))
    f.write("Detection 5 Number Summary: {:s} [Max Boxes]\n".format(n_txt))
    
    ndam_txt = " ".join(map(str, np.round(r['ndam_fivesum'], 2).tolist()))
    f.write("Damage 5 Number Summary: {:s}\n".format(ndam_txt))

    damaged_ratio = r['ndamaged']/r['ndetect']
    f.write("Class Weighting (damaged): {:.4f}\n\n".format(damaged_ratio)) 
    
    # > Features

    occ_text = " ".join(map(str, np.round(r['occ_fivesum'], 2).tolist()))
    f.write("Occlusion 5 Number Summary: {:s}\n".format(occ_text))    
    
    tru_txt = " ".join(["({:.1f}) {:d}".format(tru_lvl, int(freq)) for freq, tru_lvl in zip(r['tru_freq'], np.arange(0.0, 2.5, 0.5))])
    f.write("Truncation Frequency: {:s}\n".format(tru_txt))    
    
    cls_txt = " ".join(["{:s} {:d}".format(VEHICLE_CLASSES[i], int(v)) for i, v in enumerate(r['cls_freq'])])
    f.write("Vehicle Class Frequency: {:s}\n\n".format(cls_txt))

    # > Max depth

    f.write("Maximum Depth: {:.6f}\n\n".format(r['max_depth']))

    # > Dimension anchor

    dim_txt = " ".join(map(str, np.round(r['median_dims'], 6).tolist()))
    f.write("Median Vehicle Dimensions: {:s}\n\n".format(dim_txt))


    # > Environment

    wthr_txt = " ".join(["{:s} {:d}".format(WEATHER_CLASSES[i], int(v)) for i, v in enumerate(r['wthr_freq'])])
    f.write("Weather Class Frequency: {:s}\n".format(wthr_txt))

    rain_txt = " ".join(map(str, np.round(r['rain_fivesum'], 2).tolist()))
    f.write("Rain 5 Number Summary (median if rain): {:s} ({:.2f})\n".format(rain_txt, r['median_ifrain']))

    snow_txt = " ".join(map(str, np.round(r['snow_fivesum'], 2).tolist()))
    f.write("Snow 5 Number Summary (median if snow): {:s} ({:.2f})\n\n".format(snow_txt, r['median_ifsnow']))
    
    # > Anchors

    f.write("Image Size: {:d}\n\n".format(image_size))

    cluster_range = [i[0] for i in r['centroids_lst']]
    std_anchor_dict = {k: [] for k in cluster_range}

    for nclusters, centroids, avg_iou in r['centroids_lst']: 
        std_anchor_dict[nclusters] = write_anchors_to_file(f, nclusters, centroids, avg_iou, image_size)

    f.close()

    print("Written exploratory results to {:s}".format(output_path))

    # Update cfg

    if cfg is not None:

        istiny = (cfg.BACKBONE[-1] == "T")
        anchors = std_anchor_dict[6] if istiny else std_anchor_dict[9]

        setattrs(cfg, DAMAGED_RATIO = damaged_ratio,
                      MAX_GT_INSTANCES = max_boxes, 
                      ANCHORS = anchors.tolist(), 
                      DIM_ANCHOR = r['median_dims'].tolist(),
                      MAX_DEPTH = r['max_depth'])

        print("Updated cfg | exploratory analysis")
    
def write_anchors_to_file(f, ncluster, centroids, avg_iou, image_size):

    # Note: Anchors are given image relative to size

    widths = centroids[:, 0]
    sorted_indices = np.argsort(widths)
    std_anchors = centroids[sorted_indices]

    anchors = np.round(centroids*image_size).astype(int)                 
    sorted_anchors = anchors[sorted_indices]

    anchor_str_list = [str(tuple(wh)) for wh in sorted_anchors]                         # Decouple (w, h) pairs
    anchor_str = ",".join(anchor_str_list)
    f.write("{:d} [{:s}] AvgIOU:{:.2f}\n".format(ncluster, anchor_str, avg_iou))        # Write (w, h) pair list to file

    return std_anchors 

# PART D: Summarising operations =======================================================

def explore(data_dir, image_size, nclusters = 14, cfg = None):

    res_dict = process_data(data_dir, image_size, nclusters)
    write_explore(data_dir, res_dict, image_size, cfg = cfg)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', default = './data/', type = str, help='training data path / output directory')
    parser.add_argument('-num_clusters', default=20, type = int, help='range of clusters')
    parser.add_argument('-image_size', default=416, help="anchor scaling dimensions")
   
    args = parser.parse_args()
    explore(args.data_path, args.image_size, args.num_clusters)

if __name__=="__main__":
    main(sys.argv)

