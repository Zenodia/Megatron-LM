#!/bin/bash 
####### not working need tweaking
EXP_NAME="MegatronWP32kBert_Svenska"
 # ngc args
INSTANCE="dgx1v.32g.2.norm"
IMAGE="nvcr.io/nvidia/pytorch:20.11-py3"
# wandb args
PROJECT_NAME=MegatronWP32kBert_Svenska
# megatron-lm args
GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
DATA_PATH=/raid/SV32k_Bert__text_sentence
CHECKPOINT_PATH=/result
VOCAB_FILE=/mnt/dataset/wp/SV_HFWordPiece_vocab32k-vocab.txt

MP_SIZE=1
DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} --nnodes ${NNODES} --node_rank ${NODE_RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"

CMD="python -m torch.distributed.launch ${DISTRIBUTED_ARGS} \
    pretrainBPEbert.py \
        --pipeline-model-parallel-size 1 \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --micro-batch-size 4 \
        --seq-length 512 \
        --max-position-embeddings 512 \
        --save ${CHECKPOINT_PATH} \
        --load ${CHECKPOINT_PATH} \
        --data-path ${DATA_PATH} \
        --vocab-file ${VOCAB_FILE} \
        --merge-file ${MERGE_FILE} \
        --data-impl mmap \
        --split 949,50,1 \
        --distributed-backend nccl \
        --DDP-impl torch \
        --lr 0.0001 \
        --lr-decay-iters 990000 \
        --train-iters 2000000 \
        --min-lr 0.00001 \
        --lr-warmup-fraction 0.01 \
        --log-interval 100 \
        --save-interval 100000 \
        --eval-interval 100000 \
        --eval-iters 1000 \
        --fp16 \
        --tensorboard-dir /result "
echo "${CMD}"
ngc batch run \
--name ${EXP_NAME} --preempt RUNONCE --ace nv-us-west-2 \
--instance ${INSTANCE} \
--commandline "nvidia-smi && \
cp -r /mnt/dataset/bpe /raid && \
cp /mnt/dataset/SV32k_Bert__text_sentence.bin /raid/ && \
cp /mnt/dataset/SV32k_Bert__text_sentence.idx /raid/ && \
ls /raid && \
git clone https://github.com/Zenodia/Megatron-LM.git && \
cd Megatron-LM/ && \
git checkout svenska && \
bash 0b_pip_install.sh && \
${CMD}" \
--result /result \
--image ${IMAGE} \
--org nvidian \
--datasetid 79057:/mnt/dataset \
--port 6006
