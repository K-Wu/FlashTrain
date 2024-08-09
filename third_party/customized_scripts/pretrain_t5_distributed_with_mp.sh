#!/bin/bash
# Adapted from https://github.com/microsoft/Megatron-DeepSpeed/blob/94dbfd1cd35fea44f3a504722060bee4962816e8/examples/pretrain_t5_distributed_with_mp.sh (forked from github.com/microsoft/Megatron-DeepSpeed/) with dataset and tokernizer path changed according to the working datasets specified in test_load_datasets.py
set -ex

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))



ENCODER_NUM_LAYERS=${ENCODER_NUM_LAYERS:-2}
DECODER_NUM_LAYERS=${DECODER_NUM_LAYERS:-2}
HIDDEN_SIZE=${HIDDEN_SIZE:-8192}
NUM_ATTN_HEADS=${NUM_ATTN_HEADS:-128}
SEQ_LENGTH=${SEQ_LENGTH:-1024}
ACTIVATION_CHECKPOINT="${ACTIVATION_CHECKPOINT:-false}" # selective, full, false
USE_TENSOR_CACHE="${USE_TENSOR_CACHE:-true}"
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-16}
TC_LOGGING_LEVEL="${TC_LOGGING_LEVEL:-CRITICAL}"

CHECKPOINT_PATH=./tmp
VOCAB_FILE=$HOME/.cache/my_huggingface_datasets/bert-base-uncased-vocab.txt
DATA_PATH="$HOME/.cache/my_huggingface_datasets/meg-bert_text_document"

T5_ARGS=""
if [ "${USE_TENSOR_CACHE}" = "true" ]; then
  T5_ARGS="${T5_ARGS} --enable-tensor-cache --tensor-cache-log-level ${TC_LOGGING_LEVEL} --cufile-malloc-hook-is-used"
fi

if [ "${ACTIVATION_CHECKPOINT}" = "selective" ]
then
    T5_ARGS="${T5_ARGS} --recompute-granularity selective --recompute-num-layers 1 --recompute-method uniform "
elif [ "${ACTIVATION_CHECKPOINT}" = "full" ] 
then
    T5_ARGS="${T5_ARGS} --recompute-granularity full --recompute-num-layers 1 --recompute-method uniform "
fi


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
#    --ffn-hidden-size 3072 \
#    --profile-first-iter-longer \
#    --tensor-cache-in-memory-adapter \
T5_ARGS="${T5_ARGS} \
    --optimizer sgd \
    --no-bias-gelu-fusion \
    --ends-on 12\
    --lossy-offload-first-iter \
    --use-pure-low-precision \
    --use-flash-attn-v2 \
    --tensor-model-parallel-size 2 \
    --encoder-num-layers ${ENCODER_NUM_LAYERS} \
    --decoder-num-layers ${DECODER_NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTN_HEADS} \
    --kv-channels 64 \
    --encoder-seq-length ${SEQ_LENGTH} \
    --decoder-seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --lr 0.0001 \
    --train-iters 1000 \
    --lr-decay-iters 1000 \
    --lr-decay-style linear \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16  \
    --fp16-lm-cross-entropy \
    --vocab-extra-ids 100
"

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

# /usr/local/cuda-12.1/bin/nsys profile -o pretrain_t5_distributed --force-overwrite true  --trace=cuda,nvtx --sample=cpu --cuda-memory-usage true \
torchrun $DISTRIBUTED_ARGS "$SCRIPTDIR"/../Megatron-DeepSpeed/pretrain_t5.py \
    $T5_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
