#!/bin/bash 
EXP_NAME="Swedish_WordPiece_32k_originalMegatron"
 # ngc args
INSTANCE="dgx1v.32g.8.norm"
IMAGE="nvcr.io/nvidia/pytorch:20.11-py3"
# wandb args
PROJECT_NAME=Swedish_WordPiece_32k_originalMegatron

# megatron-lm args
GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
DATA_PATH=/raid/dataset/zhvocab32000_train_chinese_sentence
CHECKPOINT_PATH=/result
VOCAB_FILE=/dataset/wp/SV_HFWordPiece_vocab32k-vocab.txt
DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} --nnodes ${NNODES} --node_rank ${NODE_RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"
CMD="python -m torch.distributed.launch ${DISTRIBUTED_ARGS} \
    pretrain_bert.py \
        --model-parallel-size 1 \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --batch-size 4 \
        --seq-length 512 \
        --max-position-embeddings 512 \
        --train-iters 2000000 \
        --save ${CHECKPOINT_PATH} \
        --load ${CHECKPOINT_PATH} \
        --data-path ${DATA_PATH} \
        --vocab-file ${VOCAB_FILE} \
        --data-impl mmap \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr 0.0001 \
        --lr-decay-style linear \
        --min-lr 1.0e-5 \
        --lr-decay-iters 990000 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --warmup .01 \
        --log-interval 100 \
        --save-interval 10000 \
        --eval-interval 1000 \
        --eval-iters 10 \
        --fp16 \
        --tensorboard-dir /result "
echo "${CMD}"
ngc batch run \
--name ${EXP_NAME} --preempt RUNONCE --ace nv-us-west-2 \
--instance ${INSTANCE} \
--commandline "nvidia-smi && \
cp -r /mount/dataset /raid && \
git clone https://github.com/Zenodia/NV_megatron.git && \
cd NV_megatron/ && \
bash 0b_pip_install.sh && \
python setup.py install && \
${CMD}" \
--result /result \
--image ${IMAGE} \
--org nvidian \
--datasetid 71028:/mount/dataset \
--port 6006