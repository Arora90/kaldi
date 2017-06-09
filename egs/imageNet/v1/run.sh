#!/bin/bash

stage=0

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

if [ $stage -le 0 ]; then
  # data preparation
  local/prepare_data.sh
  for x in {train_t12,train_t3,test}; do
    image/validate_image_dir.sh data/$x
  done
fi


#image/nnet3/get_egs.sh --cmd "$train_cmd" data/train_t12 data/test exp/train_t12_egs
#image/nnet3/get_egs.sh --cmd "$train_cmd" data/train_t3 data/test exp/train_t3_egs


# prepare a different version of the egs with 2 instead of 3 archives.
#image/nnet3/get_egs.sh --egs-per-archive 30000 --cmd "$train_cmd" data/train_t12 data/test exp/train_t12_egs2
#image/nnet3/get_egs.sh --egs-per-archive 30000 --cmd "$train_cmd" data/train_t3 data/test exp/train_t3_egs2

