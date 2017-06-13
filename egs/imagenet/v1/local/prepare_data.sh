#!/bin/bash

# copyright 2017 Johns Hopkins University
# Apache 2.0

# This script loads the development kit training validation and test data for Imagenet 2012 task 1,2 and 3.

[ -f ./path.sh ] && . ./path.sh; # source the path.

dl_dir=data/download
devkit_t12=$dl_dir/devkit_t12
devkit_t12_url=http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz
devkit_t3=$dl_dir/devkit_t3
devkit_t3_url=http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t3.tar.gz

train_t12=$dl_dir/train_t12
train_t12_url=http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar
train_t3=$dl_dir/train_t3
train_t3_url=http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train_t3.tar

val_data=$dl_dir/test
val_data_url=http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar
test_data=$dl_dir/test2_wo_labels
test_data_url=http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar
ifAugmentTestData = true 

mkdir -p $dl_dir
#download and extract development kit for task 1,2, and 3
if [ -d $devkit_t12 ]; then
  echo Not downloading Development kit, Task 1 and 2 as it is already there.
else
  if [ ! -f $dl_dir/ILSVRC2012_devkit_t12.tar.gz ]; then
    echo Downloading Development kit Task 1 and 2...
    wget -P $dl_dir $devkit_t12_url || exit 1;
  fi
  mkdir -p $devkit_t12
  tar -xvzf $dl_dir/ILSVRC2012_devkit_t12.tar.gz -C $devkit_t12 || exit 1;
  echo Done downloading and extracting Development kit, Task 1 and 2
fi

if [ -d $devkit_t3 ]; then
  echo Not downloading Development kit, Task 3 as it is already there.
else
  if [ ! -f $dl_dir/ILSVRC2012_devkit_t3.tar.gz ]; then
    echo Downloading Development kit, Task 3...
    wget -P $dl_dir $devkit_t3_url || exit 1;
  fi
  mkdir -p $devkit_t3
  tar -xvzf $dl_dir/ILSVRC2012_devkit_t3.tar.gz -C $devkit_t3 || exit 1;
  echo Done downloading and extracting Development kit, Task 3
fi

#download and extract training data for task 1,2, and 3
if [ -d $train_t12 ]; then
  echo Not downloading Training data, Task 1 and 2 as it is already there.
else
  if [ ! -f $dl_dir/ILSVRC2012_img_train.tar ]; then
    echo Downloading Training data, Task 1 and 2...
    wget -P $dl_dir $train_t12_url || exit 1;
  fi
  mkdir -p $train_t12
  tar -xvf $dl_dir/ILSVRC2012_img_train.tar -C $train_t12 || exit 1;
  echo Done downloading and extracting Training data, Task 1 and 2
  find $train_t12 -name "*.tar" | while read NAME ;  do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}";done
fi

if [ -d $train_t3 ]; then
  echo Not downloading Training data Task 3 as it is already there.
else
  if [ ! -f $dl_dir/ILSVRC2012_img_train_t3.tar ]; then
    echo Downloading Training data, Task 3...
    wget -P $dl_dir $train_t3_url || exit 1;
  fi
  mkdir -p $train_t3
  tar -xvf $train_t3/ILSVRC2012_img_train_t3.tar -C $train_t3 || exit 1;
  echo Done downloading and extracting Training data, Task 3
  find $train_t3 -name "*.tar" | while read NAME ;  do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}";done
fi

#download and extract validation and test data
if [ -d $val_data ]; then
  echo Not downloading validation data as it is already there.
else
  if [ ! -f $dl_dir/ILSVRC2012_img_val.tar ]; then
    echo Downloading validation data...
    wget -P $dl_dir $val_data_url || exit 1;
  fi
  mkdir -p $val_data
  tar -xvf $dl_dir/ILSVRC2012_img_val.tar -C $val_data || exit 1;
  echo Done downloading and extracting validation data
fi

if [ -d $test_data ]; then
  echo Not downloading test data as it is already there.
else
  if [ ! -f $dl_dir/ILSVRC2012_img_test.tar ]; then
    echo Downloading validation data...
    wget -P $dl_dir $test_data_url || exit 1;
  fi
  mkdir -p $test_data
  tar -xvf $dl_dir/ILSVRC2012_img_test.tar -C $test_data || exit 1;
  echo Done downloading and extracting test data
fi

# create train and test directory
mkdir -p data/{train_t12,train_t3,test}/data

write_test_labels_classes.py $ifAugmentTestData
cp data/test/classes.txt data/train_t12/classes.txt
cp data/test/classes.txt data/train_t3/classes.txt

echo 3 > data/train_t12/num_channels
echo 3 > data/train_t3/num_channels
echo 3 > data/test/num_channels

local/process_test_data.py $val_data $ifAugmentTestData 256 224| \
  copy-feats --compress=true --compression-method=7 \
    ark:- ark,scp:data/test/data/images.ark,data/test/images.scp || exit 1

local/process_train_data.py $train_t12 data/train_t12/ | \
  copy-feats --compress=true --compression-method=7 \
   ark:- ark,scp:data/train_t12/data/images.ark,data/train_t12/images.scp || exit 1

local/process_train_data.py $train_t3 data/train_t3/ | \
  copy-feats --compress=true --compression-method=7 \
   ark:- ark,scp:data/train_t3/data/images.ark,data/train_t3/images.scp || exit 1

