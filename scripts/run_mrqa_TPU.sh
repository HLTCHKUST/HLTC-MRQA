#!/bin/bash

#### local path
DATA_DIR=/home/hltcmrqa/mrqa/mrqa-data/data
LOCAL_DIR=/home/hltcmrqa/mrqa/mrqa-data/model

#### google storage path
GS_ROOT=gs://mrqa-data
GS_INIT_CKPT_DIR=${GS_ROOT}/model
GS_PROC_DATA_DIR=${GS_ROOT}/proc_data/mrqa
GS_MODEL_DIR=${GS_ROOT}/experiment/mrqa
GS_PREDICT_DIR=${GS_ROOT}/result/mrqa

# TPU name in google cloud
TPU_NAME=node-2

python run_multiqa.py \
  --use_tpu=True \
  --tpu=${TPU_NAME} \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --model_config_path=${GS_INIT_CKPT_DIR}/xlnet_config.json \
  --spiece_model_file=${LOCAL_DIR}/spiece.model \
  --output_dir=${GS_PROC_DATA_DIR} \
  --init_checkpoint=${GS_INIT_CKPT_DIR}/xlnet_model.ckpt \
  --model_dir=${GS_MODEL_DIR} \
  --train_dir=${DATA_DIR}/train \
  --dev_dir=${DATA_DIR}/dev \
  --uncased=False \
  --max_seq_length=512 \
  --do_train=True \
  --train_batch_size=48 \
  --do_predict=True \
  --predict_dir=${GS_PREDICT_DIR} \
  --predict_batch_size=32 \
  --learning_rate=1e-5 \
  --adam_epsilon=1e-6 \
  --iterations=1000 \
  --save_steps=1000 \
  --train_steps=10000 \
  --warmup_steps=1000 \
  $@