import argparse
import numpy as np
import sys
import os
import random 
import tensorflow as tf

from data import read_and_parse
from _data_reformat import get_bb2d 

# Description: Adapt gen_anchors.py for 
#   (a) Use on custom data
#   (b) Kmedians
#   (c) Determine max boxes for y padding

# Functions:
#    a. Read in data
#    b. Calculate kmeans anchors [1 - 20]
#    c. Write anchors to file

# A: Read in data and convert bb (xmin, ymin, xmax, ymax) to standardised anchors (w, h) 

def extract_wh(x, cpos, features, camera, environment):
    bb2d = get_bb2d(camera, cpos)
    bb2d = tf.transpose(bb2d)
    box_wh = bb2d[:, 2:4] - bb2d[:, :2] 
    return box_wh

def extract_n(x, cpos, features, camera, environment):
    N = tf.gather(tf.shape(features), 0)
    
    dams = features[:, 0]
    ndam = tf.reduce_sum(dams)
    return [N, ndam]   

# B: Calculate kmeans anchors

def calculate_anchors(box_wh, nclusters, eps = 0.005):
    centroids = np.vstack(random.sample(list(box_wh), nclusters))
    return kmeans(box_wh, centroids, eps)

def IOU(x,centroids):
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

def write_anchors_to_file(f, ncluster, centroids, avg_iou, size):

    # Note: Anchors are given relative to size

    anchors = np.round(centroids*size).astype(int)                
             
    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    anchor_str_list = [str(tuple(wh)) for wh in anchors[sorted_indices]]                 # Decouple (w, h) pairs
    anchor_str = ",".join(anchor_str_list)
    f.write("{:d} [{:s}] AvgIOU:{:.2f}\n".format(ncluster, anchor_str, avg_iou)) # Write (w, h) pair list to file

    print('ncluster: {:d} [{:s}]'.format(ncluster, anchor_str))  
   
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

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', default = './data/', type = str, help='training data path')
    parser.add_argument('-output_dir', default = './data/', type = str, help='output anchor directory')  
    parser.add_argument('-num_clusters', default = 0, type = int, help='number of clusters')
    parser.add_argument('-size', default=416, help="anchor scaling dimensions")
   
    args = parser.parse_args()
    
    # A: Read in data and convert bb (xmin, ymin, xmax, ymax) to standardised anchors (w, h) 

    ds = read_and_parse(args.data_path, size=args.size, istrain=True) # [x, cpos, features, camera, environment]
    ds_wh = ds.map(extract_wh)                                        # [width, height]
    box_wh = np.vstack(list(ds_wh))

    ds_n = ds.map(extract_n)                                          # [N, ndam] # Modify: Can add truncation, occ dbn
    arr_n = np.array(list(ds_n)) 
    ndetect = int(np.sum(arr_n[:, 0]))
    ndamaged = int(np.sum(arr_n[:, 1]))
    max_boxes = int(np.max(arr_n[:, 0]))

    # B/C: Calculate mediod anchors and write to file

    output_path = os.path.join(args.output_dir, 'drs_anchors.txt')

    f = open(output_path, 'w')
    f.write("Number of Detections (damaged): {:d} ({:d})\n".format(ndetect, ndamaged))
    f.write("Class Weighting (damaged): {:.4f}\n".format(ndamaged/ndetect))
    f.write("Max Boxes: {:d}\n".format(max_boxes))    
    f.write("Size: {:d}\n".format(args.size))

    if args.num_clusters == 0:
        for nclusters in range(1, 15): # CHANGE   
            anchors = calculate_anchors(box_wh, nclusters)
            avg_iou = avg_IOU(box_wh, anchors)
            write_anchors_to_file(f, nclusters, anchors, avg_iou, args.size)
            print('nclusters:', nclusters, " avg iou:", avg_iou)
    else:
        anchors = calculate_anchors(box_wh, args.num_clusters)
        avg_iou = avg_IOU(box_wh, anchors)
        write_anchors_to_file(f, nclusters, anchors, avg_iou, args.size)
        print('nclusters:', nclusters, " avg iou:", avg_iou)
        
    f.close()

if __name__=="__main__":
    main(sys.argv)
