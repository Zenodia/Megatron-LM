WORLD_SIZE=2
MP_SIZE=2
#PIPELINE_SIZE=2

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT_PATH=./zh_ckpt
VOCAB_FILE=./cn/bpe/32k/vocab.json
MERGE_FILE=./cn/bpe/32k/merges.txt
DATA_PATH=./dataset/demo_bert_text_sentence
BERT_ARGS="--num-layers 24 \
           --hidden-size 1024 \
           --num-attention-heads 16 \
           --seq-length 512 \
           --max-position-embeddings 512 \
           --lr 0.0001 \
           --train-iters 1000000 \
           --min-lr 0.00001 \
           --lr-decay-iters 990000 \
           --lr-warmup-fraction 0.01 \
           --micro-batch-size 64 \
           --vocab-file $VOCAB_FILE \
           --merge-file $MERGE_FILE\
           --split 949,50,1 \
           --fp16"
OUTPUT_ARGS="--log-interval 10 \
             --save-interval 500 \
             --eval-interval 100 \
             --eval-iters 10 \
             --checkpoint-activations"

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./Zeno_modifed_pretrainBPEbert.py \
                $BERT_ARGS \
                $OUTPUT_ARGS \
                --save $CHECKPOINT_PATH \
                --load $CHECKPOINT_PATH \
                --data-path $DATA_PATH \
                --DDP-impl torch 
                --tensor-model-parallel-size $MP_SIZE \
                #--pipeline-model-parallel-size $PIPELINE_SIZE \

