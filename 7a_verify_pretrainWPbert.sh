CHECKPOINT_PATH=/workspace/SVdata/sv_ckpt/pretrained/wp_ckpt/
VOCAB_FILE=./SVdata/ngc/SVCC100sprakbank_HFWP32k-vocab.txt

DATA_PATH=./SVdata/ngc/SVCC100_sprakbank_32kWP_text_sentence
## bert large configuration

BERT_ARGS="--num-layers 24 \
           --hidden-size 1024 \
           --num-attention-heads 16 \
           --seq-length 512 \
           --max-position-embeddings 512 \
           --lr 0.0001 \
           --train-iters 4000000 \
           --min-lr 0.00001 \
           --lr-decay-iters 9990000 \
           --lr-warmup-fraction 0.01 \
           --override-lr-scheduler \
           --micro-batch-size 8 \
           --vocab-file $VOCAB_FILE \
           --split 949,50,1 \
           --fp16"

OUTPUT_ARGS="--log-interval 1000 \
             --save-interval 250000 \
             --eval-interval 250000 \
             --eval-iters 50000 \
             --checkpoint-activations"

python pretrainWPbert.py \
       --tensor-model-parallel-size 1 \
       $BERT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH