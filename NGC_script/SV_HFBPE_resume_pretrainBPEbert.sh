#!/bin/bash 
####### not working need tweaking
EXP_NAME="MegatronBPE32kBert_Svenska"
 # ngc args
INSTANCE="dgx1v.32g.4.norm"
IMAGE="nvcr.io/nvidia/pytorch:20.11-py3"
# wandb args
PROJECT_NAME=MegatronBPE32kBert_Svenska
# megatron-lm args
GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
DATA_PATH=/raid/SV32k_Bert__text_sentence
CHECKPOINT_PATH=/result
VOCAB_FILE=/mnt/dataset/bpe/32k/vocab.json
MERGE_FILE=/mnt/dataset/bpe/32k/merges.txt
MP_SIZE=1
DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} --nnodes ${NNODES} --node_rank ${NODE_RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"
BERT_ARGS="--num-layers 24 \
           --hidden-size 1024 \
           --num-attention-heads 16 \
           --seq-length 512 \
           --max-position-embeddings 512 \
           --lr 0.0001 \
           --train-iters 8000000 \
           --min-lr 0.00001 \
           --lr-decay-iters 990000 \
           --lr-warmup-fraction 0.01 \
           --micro-batch-size 8 \
           --override-lr-scheduler \
           --vocab-file ${VOCAB_FILE} \
           --merge-file ${MERGE_FILE}\
           --split 949,50,1 \
           --fp16"

OUTPUT_ARGS="--log-interval 10000 \
             --save-interval 10000 \
             --eval-interval 10000 \
             --eval-iters 100 \
             --checkpoint-activations"
     
CMD="python -m torch.distributed.launch ${DISTRIBUTED_ARGS} \
    pretrainBPEbert.py \
        --tensor-model-parallel-size 1 \
       ${BERT_ARGS} \
       ${OUTPUT_ARGS} \
       --save ${CHECKPOINT_PATH} \
       --load ${CHECKPOINT_PATH} \
       --data-path ${DATA_PATH}
       --tensorboard-dir /result "
echo "${CMD}"
ngc batch run \
--name ${EXP_NAME} --preempt RUNONCE --ace nv-us-west-2 \
--instance ${INSTANCE} \
--commandline "nvidia-smi && \
cp -r /mnt/dataset/bpe /raid && \
cp /mnt/dataset/SV32k_Bert__text_sentence.bin /raid/ && \
cp /mnt/dataset/SV32k_Bert__text_sentence.idx /raid/ && \
cp -r /mount/ckpt/iter_2211500 /result && \
cp /mount/ckpt/latest_checkpointed_iteration.txt /result && \
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
--datasetid 79950:/mount/ckpt \
--port 6006
