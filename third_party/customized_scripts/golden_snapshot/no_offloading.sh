#!/bin/bash
# Adapted from https://github.com/microsoft/Megatron-DeepSpeed/blob/94dbfd1cd35fea44f3a504722060bee4962816e8/examples/pretrain_bert_distributed_with_mp.sh (forked from github.com/microsoft/Megatron-DeepSpeed/) with dataset and tokernizer path changed according to the working datasets specified in test_load_datasets.py
# Incorporated arguments from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/examples_deepspeed/data_efficiency/bert/pretrain/ds_pretrain_bert_336M_base_script.sh#L13
set -ex

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


# Change the below configurations here
BASE_PATH=./tmp
# Create the directory in order to save the deepspeed.json file
mkdir -p $BASE_PATH
DS_CONFIG=${BASE_PATH}/deepspeed.json

CHECKPOINT_PATH=./tmp
VOCAB_FILE=$HOME/.cache/my_huggingface_datasets/bert-base-uncased-vocab.txt
DATA_PATH="$HOME/.cache/my_huggingface_datasets/meg-bert_text_document"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

## In addition, we find that the 3.9B model (even after tuning INIT_STD) has
## NaN loss issue from the beginning thus unable to train. This is probably
## because in this example we use the public Pile data, which is a more diverse
## (and potentially more noisy) data than what used in Megatron paper. One
## potential solution is only use the sub datasets in Pile that are also
## used by Megatron paper.

## BERT 110M (same config as original BERT-Base model)
## This config is not included in Megatron-LM paper
# NUM_LAYERS=12
# HIDDEN_SIZE=768
# NUM_ATTN_HEADS=12
# INIT_STD=0.02

## BERT 336M (same config as original BERT-Large model)
#NUM_LAYERS=24
# HIDDEN_SIZE=1024
#NUM_ATTN_HEADS=16
# INIT_STD=0.02

## BERT 1.3B
# NUM_LAYERS=24
# HIDDEN_SIZE=2048
# NUM_ATTN_HEADS=32
# INIT_STD=0.013

## BERT 3.9B
# NUM_LAYERS=48
# HIDDEN_SIZE=2560
# NUM_ATTN_HEADS=40
# INIT_STD=0.011

## My custom BERT
# NUM_LAYERS=3
# HIDDEN_SIZE=8192
# HIDDEN_SIZE=6144
# NUM_ATTN_HEADS=32
# NUM_ATTN_HEADS=64

NUM_LAYERS=${NUM_LAYERS:-5}
HIDDEN_SIZE=${HIDDEN_SIZE:-8192}
NUM_ATTN_HEADS=${NUM_ATTN_HEADS:-32}
SEQ_LENGTH=${SEQ_LENGTH:-1024}
ACTIVATION_CHECKPOINT="${ACTIVATION_CHECKPOINT:-false}" # selective, full, false
USE_TENSOR_CACHE="${USE_TENSOR_CACHE:-false}"
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-16}
TC_LOGGING_LEVEL="${TC_LOGGING_LEVEL:-CRITICAL}"

ZERO_STAGE=0
INIT_STD=0.02

LTD_ENABLED="false"


ENABLE_DEEPSPEED="false"
ds_args=""
if [ "$ENABLE_DEEPSPEED" = "true" ]; then
cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true
  }
}
EOT

  ds_args=" --deepspeed ${ds_args}"
  ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
  ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"

  if [ "${ACTIVATION_CHECKPOINT}" = "selective" ] || [ "${ACTIVATION_CHECKPOINT}" = "full"  ]
  then
    ds_args="--deepspeed-activation-checkpointing ${ds_args}"
  fi
fi


BERT_ARGS=""
#  --tensor-cache-log-level CRITICAL
# --tensor-cache-in-memory-adapter 
if [ "${USE_TENSOR_CACHE}" = "true" ]; then
  BERT_ARGS="${BERT_ARGS} --enable-tensor-cache --tensor-cache-log-level ${TC_LOGGING_LEVEL} --cufile-malloc-hook-is-used"
fi

if [ "${ACTIVATION_CHECKPOINT}" = "selective" ]
then
    BERT_ARGS="${BERT_ARGS} --recompute-granularity selective --recompute-num-layers 1 --recompute-method uniform "
elif [ "${ACTIVATION_CHECKPOINT}" = "full" ] 
then
    BERT_ARGS="${BERT_ARGS} --recompute-granularity full --recompute-num-layers 1 --recompute-method uniform "
fi



# --profile-memory-beginning \
# --profile-first-iter \
# --profile-first-iter-longer \
# --optimizer sgd\
# --use-distributed-optimizer \

BERT_ARGS="${BERT_ARGS} \
    --profile-memory-beginning \
    --disable-adaptive-keep \
    --disable-adaptive-keep-passive \
    --optimizer sgd\
    --ends-on 12\
    --lossy-offload-first-iter \
    --use-pure-low-precision \
    --use-flash-attn-v2 \
    --no-bias-gelu-fusion \
    --bert-no-binary-head \
    --tensor-model-parallel-size 2 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTN_HEADS} \
    --init-method-std ${INIT_STD} \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --lr 0.0001 \
    --train-iters 1000 \
    --lr-decay-iters 990 \
    --lr-decay-style linear \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --fp16-lm-cross-entropy \
"


# Do not use --checkpoint-activations in any case. It is superceded by --deepspeed-activation-checkpointing and --recompute-granularity
# if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
# BERT_ARGS="${BERT_ARGS} \
#     --checkpoint-activations"
# fi

if [ "${LTD_ENABLED}" = "true" ]; then
BERT_ARGS="${BERT_ARGS} \
    --attention-dropout ${dropout} \
    --hidden-dropout ${dropout} \
    --random-ltd"
fi


DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 100
"

SCRIPTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd )"

export KVIKIO_COMPAT_MODE=0
export LD_PRELOAD=/home/kunwu2/FlashTrain/flashtrain/malloc_hook/hook.so

# /usr/local/cuda-12.1/bin/nsys profile -o pretrain_bert_distributed --force-overwrite true  --trace=cuda,nvtx --sample=cpu --cuda-memory-usage true \
torchrun $DISTRIBUTED_ARGS "$SCRIPTDIR"/../../Megatron-DeepSpeed/pretrain_bert.py \
    $BERT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    $ds_args
