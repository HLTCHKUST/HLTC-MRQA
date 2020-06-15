#!/bin/bash

#### local path
DATA_DIR=data
INIT_CKPT_DIR=model/xlnet_cased_L-24_H-1024_A-16
PROC_DATA_DIR=data/proc_data/mrqa
MODEL_DIR=experiment/mrqa

#### Preprocess
python run_mrqa_GPU.py \
  --use_tpu=False \
  --do_prepro=True \
  --spiece_model_file=${INIT_CKPT_DIR}/spiece.model \
  --train_dir=${DATA_DIR}/train \
  --output_dir=${PROC_DATA_DIR} \
  --uncased=False \
  --max_seq_length=340 \
  $@

# #### Potential multi-processing version
# NUM_PROC=8
# for i in `seq 0 $((NUM_PROC - 1))`; do
#   python run_multiqa_GPU.py \
#     --use_tpu=False \
#     --do_prepro=True \
#     --spiece_model_file=${INIT_CKPT_DIR}/spiece.model \
#     --train_dir=${DATA_DIR}/train \
#     --output_dir=${PROC_DATA_DIR} \
#     --uncased=False \
#     --max_seq_length=512 \
#     --num_proc=${NUM_PROC} \
#     --proc_id=${i} \
#     $@ &
# done