#!/bin/bash 
EXP_NAME="Svenska_32k_GPT_megatron"
 # ngc args
INSTANCE="dgx1v.32g.8.norm"
IMAGE="nvcr.io/nvidia/pytorch:20.11-py3"
# wandb args
PROJECT_NAME=Svenska_32k_GPT_megatron
# megatron-lm args
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
DATA_PATH=/raid/dataset/SV32k_GPT__text_sentence
CHECKPOINT_PATH=/result
VOCAB_FILE=/mnt/dataset/bpe/32k/vocab.json
MERGE_FILE=/mnt/dataset/bpe/32k/merges.txt
DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} --nnodes ${NNODES} --node_rank ${NODE_RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"

CMD="python -m torch.distributed.launch ${DISTRIBUTED_ARGS} \
    pretrain_gpt.py \
        --model-parallel-size 1 \
        --DDP-impl torch \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --batch-size 1 \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --train-iters 5000000 \
        --save ${CHECKPOINT_PATH} \
        --load ${CHECKPOINT_PATH} \
        --data-path ${DATA_PATH} \
        --vocab-file ${VOCAB_FILE} \
        --merge-file ${MERGE_FILE} \
        --distributed-backend nccl \
        --lr 0.00015 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --lr-decay-iters 320000 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --warmup .01 \
        --log-interval 1 \
        --save-interval 100000 \
        --eval-interval 100000 \
        --eval-iters 10 \
        --fp16 \
        --tensorboard-dir /result "
echo "${CMD}"
ngc batch run \
--name ${EXP_NAME} --preempt RUNONCE --ace nv-us-west-2 \
--instance ${INSTANCE} \
--commandline "nvidia-smi && \
cp -r /mnt/dataset /raid && \
git clone https://github.com/Zenodia/Megatron-LM.git && \
cd Megatron-LM/ && \
git checkout svenska && \
bash 0b_pip_install.sh && \
python setup.py install && \
${CMD}" \
--result /result \
--image ${IMAGE} \
--org nvidian \
--datasetid :/mnt/dataset \
--port 6006
