#!/bin/bash


#SBATCH -A swdl -N 2 -p interactive --job-name=swdl-gpt3:gpt3_interleave_43B_2N_16A100 --ntasks-per-node=8

set -eux

DIR='/megatron_workspace'
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
CHECKPOINT_PATH=$DIR/sv_gpt3_ckpt/
VOCAB_FILE=$DIR/32k/vocab.json
MERGE_FILE=$DIR/32k/merges.txt
DATA_PATH=$DIR/SV_CC100Sprakbank_text_document

NHIDDEN=6144
NLAYERS=96
NHEADS=96
SEQ_LEN=1024
VOCAB_SIZE=32000
MODEL_SIZE=$((($NLAYERS * (12*$NHIDDEN**2 + 13*$NHIDDEN) + ($VOCAB_SIZE * $NHIDDEN) + ($SEQ_LEN * $NHIDDEN) ) / 10**9))
EXACT_MODEL_SIZE=$(($NLAYERS * (12*$NHIDDEN**2 + 13*$NHIDDEN) + ($VOCAB_SIZE * $NHIDDEN) + ($SEQ_LEN * $NHIDDEN) ))


options="--num-layers 96 \
                --hidden-size 6144 \
                --num-attention-heads 96 \
                --seq-length 1024 \
                --max-position-embeddings 1024 \
                --lr 0.00015 \
                --train-iters 100 \
                --min-lr 0.00001 \
                --lr-decay-iters 99 \
                --lr-warmup-fraction 0.01 \
                --override-lr-scheduler \
                --micro-batch-size 1 \
                --global-batch-size 32 \
                --vocab-file ${VOCAB_FILE} \
                --merge-file ${MERGE_FILE} \
                --split 949,50,1 \
                --distributed-backend nccl \
                --log-interval 10 \
                --save-interval 100 \
                --eval-interval 100 \
                --eval-iters 100 \
                --checkpoint-activations \
                --tensor-model-parallel-size 4 \
                --pipeline-model-parallel-size 4 \
                --num-layers-per-virtual-pipeline-stage 2 \
                --save ${CHECKPOINT_PATH} \
                --load ${CHECKPOINT_PATH} \
                --data-path ${DATA_PATH} \
                --fp16 "

run_cmd="python -u ${DIR}/Megatron-LM/pretrain_gpt.py $@ ${options}"


srun -A swdl -N 2 --job-name=swdl-gpt3:SV_interleave_43B_2N_16A100 --ntasks-per-node=8 \
     --container-image "nvcr.io#nvidia/pytorch:20.12-py3" \
     --container-mounts "/lustre/fsw/swdl/zcharpy/:/megatron_workspace" \
     --output=//lustre/fsw/swdl/zcharpy/sv_gpt3_ckpt/interleave_43B_%x_%j_$DATETIME.log sh -c "${run_cmd}"
echo $MODEL_SIZE
echo $EXACT_MODEL_SIZE

set +x