#!/bin/bash
# Adapted from https://github.com/microsoft/Megatron-DeepSpeed/blob/94dbfd1cd35fea44f3a504722060bee4962816e8/examples/pretrain_bert_distributed_with_mp.sh (forked from github.com/microsoft/Megatron-DeepSpeed/) with dataset and tokernizer path changed according to the working datasets specified in test_load_datasets.py

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

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

BERT_ARGS="
    --bert-no-binary-head \
    --tensor-model-parallel-size 2 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 2 \
    --global-batch-size 16 \
    --lr 0.0001 \
    --train-iters 1000 \
    --lr-decay-iters 990 \
    --lr-decay-style linear \
    --min-lr 1.0e-5 \
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
    --log-interval 10 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 10
"

torchrun $DISTRIBUTED_ARGS pretrain_bert.py \
    $BERT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
