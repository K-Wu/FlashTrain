#!/bin/bash
# Adapted from https://github.com/microsoft/Megatron-DeepSpeed/blob/94dbfd1cd35fea44f3a504722060bee4962816e8/examples/pretrain_llama2_distributed.sh (forked from github.com/microsoft/Megatron-DeepSpeed/) with dataset and tokernizer path changed according to the working datasets specified in test_load_datasets.py
# GPT-2 portion is incorporated from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/examples_deepspeed/curriculum_learning/ds_pretrain_gpt2.sh
# This example script is contributed by external user https://github.com/nrailgun
set -ex

export CUDA_DEVICE_MAX_CONNECTIONS=1


######################################
# Change the below configurations here
BASE_PATH=./tmp
# Create the directory in order to save the deepspeed.json file
mkdir -p $BASE_PATH
DS_CONFIG=${BASE_PATH}/deepspeed.json
# DATASET_1="$HOME/.cache/my_huggingface_datasets/meg-gpt2_text_document"
# DATASET="1 ${DATASET_1}"
CHECKPOINT_PATH=./tmp
TOKENIZER_PATH=$HOME/.cache/my_huggingface_datasets/tokenizer.model # offical llama tokenizer.model


VOCAB_FILE=$HOME/.cache/my_huggingface_datasets/bert-base-uncased-vocab.txt
DATASET="$HOME/.cache/my_huggingface_datasets/meg-bert_text_document"

TP=2
PP=1
ZERO_STAGE=0

GPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


NUM_LAYERS=${NUM_LAYERS:-4}
HIDDEN_SIZE=${HIDDEN_SIZE:-8192}
NUM_ATTN_HEADS=${NUM_ATTN_HEADS:-64}
SEQ_LENGTH=${SEQ_LENGTH:-1024}
ACTIVATION_CHECKPOINT="${ACTIVATION_CHECKPOINT:-false}" # selective, full, false
USE_TENSOR_CACHE="${USE_TENSOR_CACHE:-true}"
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
TC_LOGGING_LEVEL="${TC_LOGGING_LEVEL:-CRITICAL}"
NUM_KV_HEADS=8
DISABLE_ADAPTIVE_KEEP="${DISABLE_ADAPTIVE_KEEP:-false}"
DISABLE_ADAPTIVE_KEEP_PASSIVE="${DISABLE_ADAPTIVE_KEEP_PASSIVE:-false}"

USE_LLAMA_INSTEAD_OF_GPT="${USE_LLAMA_INSTEAD_OF_GPT:-false}"
hyperparam_args="--hidden-size $HIDDEN_SIZE --num-layers $NUM_LAYERS --num-attention-heads $NUM_ATTN_HEADS"



TRAIN_STEPS=2500 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1.0


llama_args="${TC_LOGGING_LEVEL}"
if [ "${USE_TENSOR_CACHE}" = "true" ]; then
  llama_args="${llama_args} --enable-tensor-cache --tensor-cache-log-level --cufile-malloc-hook-is-used"
elif [ "${USE_TENSOR_CACHE}" = "memory" ]; then
  llama_args="${llama_args} --enable-tensor-cache --tensor-cache-log-level --cufile-malloc-hook-is-used --tensor-cache-in-memory-adapter"
fi

if [ "${ACTIVATION_CHECKPOINT}" = "selective" ]
then
    llama_args="${llama_args} --recompute-granularity selective --recompute-num-layers 1 --recompute-method uniform "
elif [ "${ACTIVATION_CHECKPOINT}" = "full" ] 
then
    llama_args="${llama_args} --recompute-granularity full --recompute-num-layers 1 --recompute-method uniform "
fi

if [ "${USE_LLAMA_INSTEAD_OF_GPT}" = "true" ]; then
  llama_args="${llama_args} --no-query-key-layer-scaling"
  llama_args="${llama_args} --attention-dropout 0.1"
  llama_args="${llama_args} --hidden-dropout 0.1"
  llama_args="${llama_args} --use-rotary-position-embeddings"
  # llama_args="${llama_args} --untie-embeddings-and-output-weights"
  # llama_args="${llama_args} --swiglu"
  llama_args="${llama_args} --swiglu-ffn-size"
  # llama_args="${llama_args} --normalization rmsnorm"
  llama_args="${llama_args} --disable-bias-linear"
  llama_args="${llama_args} --num-key-value-heads $NUM_KV_HEADS" # Line addition by KWU
fi
######################################

if [ "${DISABLE_ADAPTIVE_KEEP}" = "true" ]; then
  llama_args="${llama_args} --disable-adaptive-keep"
fi
if [ "${DISABLE_ADAPTIVE_KEEP_PASSIVE}" = "true" ]; then
  llama_args="${llama_args} --disable-adaptive-keep-passive"
fi


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
  "fp16": {
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

    ## Don't use the following old argument for recomputing the transformer layer!
    # ds_args="--checkpoint-activations ${ds_args}"

    ## Use instead the new argument for recomputing the transformer layer
    ## KWU: Support to Megatron's new checkpointing mechanism is added in https://github.com/microsoft/Megatron-DeepSpeed/pull/243
    # ds_args="--recompute-granularity full --recompute-method uniform ${ds_args}"
    ## new argument for recomputing only the attention layer
    # ds_args="--recompute-granularity selective ${ds_args}"
  fi
fi


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

SCRIPTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd )"

export KVIKIO_COMPAT_MODE=0
export LD_PRELOAD=/home/kunwu2/FlashTrain/flashtrain/malloc_hook/hook.so

#      --profile-first-iter-longer \
#      --tensor-cache-in-memory-adapter \
#      --tensor-cache-reset-in-every-iteration \
# /usr/local/cuda-12.1/bin/nsys profile -o pretrain_gpt_distributed --force-overwrite true  --trace=cuda,nvtx --sample=cpu --cuda-memory-usage true \
torchrun $DISTRIBUTED_ARGS \
       "$SCRIPTDIR"/../Megatron-DeepSpeed/pretrain_gpt.py \
       --ends-on 12 \
       --optimizer sgd \
       --lossy-offload-first-iter \
       --fp16-lm-cross-entropy \
       --use-pure-low-precision \
       --use-flash-attn-v2 \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       $hyperparam_args \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters $TRAIN_STEPS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATASET \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --tokenizer-type GPTSentencePieceTokenizer \
       --tokenizer-model $TOKENIZER_PATH \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr $LR \
       --lr-decay-style cosine \
       --min-lr $MIN_LR \
       --weight-decay $WEIGHT_DECAY \
       --clip-grad $GRAD_CLIP \
       --lr-warmup-iters $LR_WARMUP_STEPS \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 10000 \
       --eval-iters 100 \
       --fp16 \
       $llama_args \
       $ds_args
