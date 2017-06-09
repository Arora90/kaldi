#!/usr/bin/env python

#copyright 2017 Johns Hopkins University
# Apache 2.0

""" This script prepares the training and test data for Imagenet 2012.
"""

import argparse
import os
import sys
import numpy as np
from scipy import misc

parser = argparse.ArgumentParser(description="""Converts train/test data of
                                                Imagenet 2012 to
                                                Kaldi feature format""")
parser.add_argument('database', type=str,
                    default='data/download/test',
                    help='path to downloaded imagenet test data')
parser.add_argument('--out-ark', type=str, default='-', help='where to write output feature data')
args = parser.parse_args()

def load_imagenet_test_data(datafile):
    data = []
    image_id = []
    directory_elements = os.listdir(datafile)
    for file in directory_elements:
        if len(file) > 5:
            if file.endswith('.JPEG'):
                file_path = os.path.join(datafile, file)
                img = misc.imread(file_path)
		temp_image_id = file.split('_')[2]  
                image_id += [temp_image_id.split('.')[0]]
		if(len(img.shape)==2):
		    img = np.repeat(img[...,None],3,axis=2)	 
                H = img.shape[0]
                W = img.shape[1]
                C = img.shape[2]
                img = np.reshape(np.transpose(img, (1, 0, 2)), (W, H * C))
                data += [img]
    return data, image_id


def write_kaldi_matrix(file_handle, matrix, key):
    # matrix is a list of lists
    file_handle.write(key + "  [ ")
    num_rows = len(matrix)
    if num_rows == 0:
        raise Exception("Matrix is empty")
    num_cols = len(matrix[0])

    for row_index in range(len(matrix)):
        if num_cols != len(matrix[row_index]):
            raise Exception("All the rows of a matrix are expected to "
                            "have the same length")
        file_handle.write(" ".join(map(lambda x: str(x), matrix[row_index])))
        if row_index != num_rows - 1:
            file_handle.write("\n")
    file_handle.write(" ]\n")


### main ###
if args.out_ark == '-':
  out_fh = sys.stdout  # output file handle to write the feats to
else:
  out_fh = open(args.out_ark, 'wb')

fpath = args.database
data, img_IDS = load_imagenet_test_data(fpath)
num_images = len(img_IDS)

for i in range(num_images):
    key = str(img_IDS[i])
    img = data[i]
    write_kaldi_matrix(out_fh, img, key)

out_fh.close()

