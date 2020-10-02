import argparse
import sys
from _data_exploratory import explore

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', default = './data/', type = str, help='training data path')
    parser.add_argument('-output_dir', default = './data/', type = str, help='output anchor directory')  
    parser.add_argument('-num_clusters', default = 14, type = int, help='range of clusters')
    parser.add_argument('-image_size', default=416, help="anchor scaling dimensions")
   
    args = parser.parse_args()
    explore(args.data_path, args.output_dir, args.image_size, args.num_clusters)

if __name__=="__main__":
    main(sys.argv)