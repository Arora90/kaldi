#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University
# Apache 2.0


""" This script reads and writes classes and labels data
"""

import scipy.io
import os

def zeropad(x, length):
  s = str(x)
  while len(s) < length:
    s = '0' + s
  return s

labels_file = os.path.join('data/test', 'labels.txt')
labels_fh = open(labels_file, 'wb')
f = open("data/download/devkit_t12/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt")

img_id = 1
for label in f.readlines():
    key = zeropad(img_id, 8)
    labels_fh.write(key + ' ' + str(label))
    img_id += 1

labels_fh.close()



meta_data = scipy.io.loadmat('data/download/devkit_t12/ILSVRC2012_devkit_t12/data/meta.mat')
classes_file = os.path.join('data/test', 'classes.txt')
classes_fh=open(classes_file, 'wb')

for i in range(1000):
    WNID_val = meta_data['synsets']['WNID'][i][0][0].encode('ascii', 'ignore')
    ILSVRC2012_ID_val = int(meta_data['synsets']['ILSVRC2012_ID'][i][0][0][0]) - 1
    val = str(WNID_val) + '  ' + str(ILSVRC2012_ID_val)
    classes_fh.write(val)
    classes_fh.write("\n")

classes_fh.close()
