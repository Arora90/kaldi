#!/usr/bin/env python

#copyright 2017 Johns Hopkins University
# Apache 2.0

""" This script prepares the training data for Imagenet 2012.
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
                    default='data/download/train_t3',
                    help='path to downloaded imagenet train data')
parser.add_argument('dir', type=str, help='output dir')
parser.add_argument('--out-ark', type=str, default='-', help='where to write output feature data')
args = parser.parse_args()


def load_imagenet_train_data(datafile, data, label_WNid):
    directory_elements = os.listdir(datafile)
    for file in directory_elements:
        if len(file) > 5:
            if file.endswith('.JPEG'):
                file_path = os.path.join(datafile, file)
                img = misc.imread(file_path)
		if(len(img.shape)==2):
                    img = np.repeat(img[...,None],3,axis=2)
                H = img.shape[0]
                W = img.shape[1]
                C = img.shape[2]
                img = np.reshape(np.transpose(img, (1, 0, 2)), (W, H * C))
                data += [img]
                label_WNid += [file.split('_')[0]]


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


def get_list_of_class_labels(arg_list_WNID):
    out_list_class_label = []
    for arg_WNID in arg_list_WNID:
        index = list_WNID.index(arg_WNID)
        arg_class_label = list_classid[index]
        out_list_class_label += [arg_class_label]
    return out_list_class_label


def zeropad(x, length):
  s = str(x)
  while len(s) < length:
    s = '0' + s
  return s


### main ###
classes_file = os.path.join(args.dir, 'classes.txt')
classes_fh = open(classes_file)
WNID_classID_mappping = classes_fh.readlines()

list_WNID = []
list_classid = []
for line_str in WNID_classID_mappping:
    list_classid += [line_str.split('  ')[0]]
    list_WNID += [line_str.split('  ')[1][0:-1]]

if args.out_ark == '-':
  out_fh = sys.stdout  # output file handle to write the feats to
else:
  out_fh = open(args.out_ark, 'wb')

fpath = args.database
data = []
WNlabels = []

directory_elements = os.listdir(fpath)
for file in directory_elements:
    if len(file) == 9:
        if file[0] == 'n':
            file_path = os.path.join(fpath, file)
            load_imagenet_train_data(file_path, data, WNlabels)

class_labels = get_list_of_class_labels(WNlabels)
num_images = np.shape(class_labels)[0]

img_id = 1
labels_file = os.path.join(args.dir, 'labels.txt')
labels_fh = open(labels_file, 'wb')

for i in range(num_images):
    key = zeropad(img_id, 8)
    img = data[i]
    write_kaldi_matrix(out_fh, img, key)
    labels_fh.write(key + ' ' + str(class_labels[i]) + '\n')
    img_id += 1

labels_fh.close()
out_fh.close()
