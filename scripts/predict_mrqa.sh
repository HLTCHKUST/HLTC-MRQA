#!/bin/bash

#### local path
DATA_DIR=data
INIT_CKPT_DIR=model/xlnet_cased_L-24_H-1024_A-16
PROC_DATA_DIR=data/proc_data/mrqa
MODEL_DIR=experiment/mrqa
PREDICT_DIR=result/mrqa

#### Use 3 GPUs, each with 8 seqlen-512 samples

python run_mrqa_GPU.py \
  --use_tpu=False \
  --num_hosts=1 \
  --num_core_per_host=1 \
  --model_config_path=${INIT_CKPT_DIR}/xlnet_config.json \
  --spiece_model_file=${INIT_CKPT_DIR}/spiece.model \
  --output_dir=${PROC_DATA_DIR} \
  --init_checkpoint=${MODEL_DIR}/model.ckpt-85000 \
  --model_dir=${MODEL_DIR} \
  --train_dir=${DATA_DIR}/train \
  --dev_dir=${DATA_DIR}/dev \
  --uncased=False \
  --max_seq_length=340 \
  --do_predict=True \
  --predict_dir=${PREDICT_DIR} \
  --predict_batch_size=1 \
  $@