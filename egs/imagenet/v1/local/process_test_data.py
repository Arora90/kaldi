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
parser.add_argument('if10cropTestData', type=bool, default = 'true')
parser.add_argument('scale', type=int, default = 256)
parser.add_argument('cropside', type=int, default = 224)
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
                curr_image_id = temp_image_id.split('.')[0]
                if len(img.shape)==2:
                    img = np.repeat(img[...,None],3,axis=2)
                if args.if10cropTestData:
                    cropped_images, ids = get_test_images(img,curr_image_id)
                    image_id = image_id + ids
                    for i in range(10):
                        img = cropped_images[i]
                        H = img.shape[0]
                        W = img.shape[1]
                        C = img.shape[2]
                        img = np.reshape(np.transpose(img, (1, 0, 2)), (W, H * C))
                        data += [img]
                else:
                    image_id += [curr_image_id]
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


def get_test_images(img, image_id):
    # get image dimensions
    H = img.shape[0]
    W = img.shape[1]
    C = img.shape[2]

    # isotropically resize image with smaller side = 256
    if (H < W):
        resize_width = int(round(float(W * scale) / float(H)))
        width_center = int(round(float(resize_width) / 2.0))
        image_resized = misc.imresize(img, (scale, resize_width, C))
        crop_image = image_resized[:, int(width_center - half_scale):int(width_center + half_scale), :]
    else:
        resize_height = int(round(float(H * scale) / float(W)))
        height_center = int(round(float(resize_height) / 2.0))
        image_resized = misc.imresize(img, (resize_height, scale, C))
        crop_image = image_resized[int(height_center - half_scale):int(height_center + half_scale), :, :]

    # 10 (224,224) crop testing
    crop1 = crop_image[:cropside, :cropside, :]
    crop2 = crop_image[(scale - cropside):, :cropside, :]
    crop3 = crop_image[(scale - cropside):, (scale - cropside):, :]
    crop4 = crop_image[:cropside, (scale - cropside):, :]
    crop5 = crop_image[(half_scale - half_crop):(half_scale + half_crop), (half_scale - half_crop):(half_scale + half_crop), :]

    crop_image_hreflection = np.flipud(crop_image)
    crop6 = crop_image_hreflection[:cropside, :cropside, :]
    crop7 = crop_image_hreflection[(scale - cropside):, :cropside, :]
    crop8 = crop_image_hreflection[(scale - cropside):, (scale - cropside):, :]
    crop9 = crop_image_hreflection[:cropside, (scale - cropside):, :]
    crop10 = crop_image_hreflection[(half_scale - half_crop):(half_scale + half_crop), (half_scale - half_crop):(half_scale + half_crop), :]

    id = []
    for i in range(10):
        id += [str((int(image_id) - 1) * 10 + i + 1)]
    cropped_images = [crop1, crop2, crop3, crop4, crop5, crop6, crop7, crop8, crop9, crop10]
    return cropped_images, id


### main ###
if args.out_ark == '-':
    out_fh = sys.stdout  # output file handle to write the feats to
else:
    out_fh = open(args.out_ark, 'wb')

cropside = int(args.cropside)
scale = int(args.scale)
half_scale = int(scale/2)
half_crop = int(cropside/2)
fpath = args.database
data, img_IDS = load_imagenet_test_data(fpath)
num_images = len(img_IDS)

for i in range(num_images):
    key = str(img_IDS[i])
    img = data[i]
    write_kaldi_matrix(out_fh, img, key)

out_fh.close()

