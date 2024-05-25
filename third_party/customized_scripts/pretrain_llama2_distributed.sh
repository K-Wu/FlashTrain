#!/bin/bash
# Adapted from https://github.com/microsoft/Megatron-DeepSpeed/blob/94dbfd1cd35fea44f3a504722060bee4962816e8/examples/pretrain_llama2_distributed.sh (forked from github.com/microsoft/Megatron-DeepSpeed/) with dataset and tokernizer path changed according to the working datasets specified in test_load_datasets.py
# GPT-2 portion is incorporated from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/examples_deepspeed/curriculum_learning/ds_pretrain_gpt2.sh
# This example script is contributed by external user https://github.com/nrailgun
set -ex

######################################
# Change the below configurations here
BASE_PATH=./tmp
# Create the directory in order to save the deepspeed.json file
mkdir -p $BASE_PATH
DS_CONFIG=${BASE_PATH}/deepspeed.json
DATASET_1="$HOME/.cache/my_huggingface_datasets/meg-gpt2_text_document"
DATASET="1 ${DATASET_1}"
CHECKPOINT_PATH=./tmp
TOKENIZER_PATH=$HOME/.cache/my_huggingface_datasets/tokenizer.model # offical llama tokenizer.model

TP=2
PP=1
ZERO_STAGE=0

GPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

USE_LLAMA_INSTEAD_OF_GPT="true"
if [ "${USE_LLAMA_INSTEAD_OF_GPT}" = "true" ]; then
  MODEL_SIZE=2048 # TODO: currently unused. Add more model sizes and use MODEL_SIZE to switch between them
else
  MODEL_SIZE=345 # 117, 345, 774, 1558 for GPT-2
fi

# Model hyperparameters
hyperparam_args=""
if [ "${USE_LLAMA_INSTEAD_OF_GPT}" = "true" ]; then
# TODO: add other model size
  HIDDEN_SIZE=2048 # e.g. llama-13b: 5120
  FFN_HIDDEN_SIZE=5504 # e.g. llama-13b: 13824
  NUM_LAYERS=24 # e.g. llama-13b: 40
  NUM_HEADS=16 # e.g. llama-13b: 40
  NUM_KV_HEADS=4 # llama2 70B uses GQA
  # Add llama-unique FFN_HIDDEN_SIZE to hyperparam_args. NUM_KV_HEADS will be added later in llama_args
  hyperparam_args="${hyperparam_args} --ffn-hidden-size $FFN_HIDDEN_SIZE"
else # GPT-2
  # 12-layer, 768-hidden, 12-heads, 117M parameters
  # 24-layer, 1024-hidden, 16-heads, 345M parameters
  # 36-layer, 1280-hidden, 20-heads, 774M parameters
  # 48-layer, 1600-hidden, 25-heads, 1558M parameters
  if [[ $MODEL_SIZE -eq 117 ]]; then
          NUM_LAYERS=12
          HIDDEN_SIZE=768
          NUM_ATTN_HEADS=12
  elif [[ $MODEL_SIZE -eq 345 ]]; then
          NUM_LAYERS=24
          HIDDEN_SIZE=1024
          NUM_ATTN_HEADS=16
  elif [[ $MODEL_SIZE -eq 774 ]]; then
          NUM_LAYERS=36
          HIDDEN_SIZE=1280
          NUM_ATTN_HEADS=20
  elif [[ $MODEL_SIZE -eq 1558 ]]; then
          NUM_LAYERS=48
          HIDDEN_SIZE=1600
          NUM_ATTN_HEADS=25
  else
          echo "Model size not supported."
          exit 1
  fi
fi
hyperparam_args="${hyperparam_args} --hidden-size $HIDDEN_SIZE --num-layers $NUM_LAYERS --num-attention-heads $NUM_HEADS"



SEQ_LENGTH=2048

MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=32 # e.g. llama: 4M tokens
TRAIN_STEPS=2500 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"
activation_checkpoint="false"

# Below configuration required for llama model as per llama paper
#  --no-query-key-layer-scaling \
#  --attention-dropout 0 \
#  --hidden-dropout 0 \
#  --use-rotary-position-embeddings \
#  --untie-embeddings-and-output-weights \
#  --swiglu \
#  --normalization rmsnorm \
#  --disable-bias-linear \
#  --num-key-value-heads $NUM_KV_HEADS # Line addition by KWU
llama_args=""

if [ "${USE_LLAMA_INSTEAD_OF_GPT}" = "true" ]; then
  llama_args="${llama_args} --no-query-key-layer-scaling"
  llama_args="${llama_args} --attention-dropout 0"
  llama_args="${llama_args} --hidden-dropout 0"
  llama_args="${llama_args} --use-rotary-position-embeddings"
  llama_args="${llama_args} --untie-embeddings-and-output-weights"
  llama_args="${llama_args} --swiglu"
  llama_args="${llama_args} --normalization rmsnorm"
  llama_args="${llama_args} --disable-bias-linear"
  llama_args="${llama_args} --num-key-value-heads $NUM_KV_HEADS"
fi
######################################



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

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"

if [ "${activation_checkpoint}" = "true" ]; then
  ds_args="--deepspeed-activation-checkpointing ${ds_args}"

  ## old argument for recomputing the transformer layer
  # ds_args="--checkpoint-activations ${ds_args}"

  ## new argument for recomputing the transformer layer
  ## KWU: Support to Megatron's new checkpointing mechanism is added in https://github.com/microsoft/Megatron-DeepSpeed/pull/243
  ds_args="--recompute-granularity full --recompute-method uniform ${ds_args}"
  ## new argument for recomputing only the attention layer
  # ds_args="--recompute-granularity selective ${ds_args}"
fi


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --enable-tensor-cache \
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
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 10000 \
       --eval-iters 10 \
       --bf16 \
       --no-query-key-layer-scaling \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization rmsnorm \
       --disable-bias-linear \
       --num-key-value-heads $NUM_KV_HEADS \
       $llama_args \
       $ds_args
