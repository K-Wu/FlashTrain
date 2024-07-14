#!/bin/bash
# Adapted from https://github.com/microsoft/Megatron-DeepSpeed/blob/94dbfd1cd35fea44f3a504722060bee4962816e8/examples/pretrain_bert.sh (forked from github.com/microsoft/Megatron-DeepSpeed/) with dataset and tokernizer path changed according to the working datasets specified in test_load_datasets.py

export CUDA_DEVICE_MAX_CONNECTIONS=1

CHECKPOINT_PATH=./tmp
VOCAB_FILE=$HOME/.cache/my_huggingface_datasets/bert-base-uncased-vocab.txt
DATA_PATH="$HOME/.cache/my_huggingface_datasets/meg-bert_text_document"
#    --num-layers 24 \
#    --hidden-size 1024 \


#    --tensor-cache-log-level CRITICAL
#    --profile-first-iter \


#    --enable-tensor-cache \
#    --tensor-cache-in-memory-adapter \


#    --deepspeed-activation-checkpointing \
#    --recompute-granularity full \
#    --recompute-num-layers 1\
#    --recompute-method uniform \



BERT_ARGS="    
    --tensor-cache-log-level ERROR \
    --enable-tensor-cache \
    --tensor-cache-in-memory-adapter \
    --use-flash-attn-v2 \
    --bert-no-binary-head \
    --num-layers 3 \
    --hidden-size 8192 \
    --num-attention-heads 16 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 8 \
    --global-batch-size 16 \
    --lr 0.0001 \
    --train-iters 1000 \
    --lr-decay-iters 990 \
    --lr-decay-style linear \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
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
    --eval-iters 10
"

SCRIPTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd )"

# According to https://forums.developer.nvidia.com/t/nsys-segfault-during-profile-pthread-create-when-connected-over-ssh-with-x11/187703/4
# /usr/local/cuda-12.1/bin/nsys profile -o pretrain_bert --force-overwrite true  --trace=cuda --sample=cpu \
torchrun  "$SCRIPTDIR"/../Megatron-DeepSpeed/pretrain_bert.py \
    $BERT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
