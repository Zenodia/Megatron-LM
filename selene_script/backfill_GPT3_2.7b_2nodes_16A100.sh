#!/bin/bash


#SBATCH -A swdl -N 2 -p backfill --job-name=sa-debug:gpt3_16A100 --ntasks-per-node=8

set -eux

DIR='/megatron_workspace'
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
CHECKPOINT_PATH=$DIR/sv_gpt3_ckpt/
VOCAB_FILE=$DIR/32k/vocab.json
MERGE_FILE=$DIR/32k/merges.txt
DATA_PATH=$DIR/SV_CC100Sprakbank_text_document


options="--num-layers 32 \
                --hidden-size 2560 \
                --num-attention-heads 32 \
                --seq-length 512 \
                --max-position-embeddings 1024 \
                --lr 0.00015 \
                --train-iters 100000 \
                --min-lr 0.00001 \
                --lr-decay-iters 99000 \
                --lr-warmup-fraction 0.01 \
                --override-lr-scheduler \
                --micro-batch-size 16 \
                --vocab-file ${VOCAB_FILE} \
                --merge-file ${MERGE_FILE} \
                --split 949,50,1 \
                --distributed-backend nccl \
                --fp16 \
                --log-interval 1000 \
                --save-interval 50000 \
                --eval-interval 50000 \
                --eval-iters 1000 \
                --checkpoint-activations \
                --tensor-model-parallel-size 2 \
                --pipeline-model-parallel-size 2 \
                --save ${CHECKPOINT_PATH} \
                --load ${CHECKPOINT_PATH} \
                --data-path ${DATA_PATH} "

run_cmd="python -u ${DIR}/Megatron-LM/pretrain_gpt.py $@ ${options}"


srun -A swdl -N 2 --job-name=swdl-gpt3:sv2.7B_multinodes_16A100 --ntasks-per-node=8 \
     --container-image "nvcr.io#nvidia/pytorch:20.12-py3" \
     --container-mounts "/lustre/fsw/swdl/zcharpy/:/megatron_workspace" \
     --output=//lustre/fsw/swdl/zcharpy/sv_gpt3_ckpt/%x_%j_$DATETIME.log sh -c "${run_cmd}"


set +x