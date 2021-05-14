CHECKPOINT_PATH=./zh_ckpt
VOCAB_FILE=./SVdata/wp/SV_HFWordPiece_vocab32k-vocab.txt

DATA_PATH=./SVdata/preprocessed/SV_WordPiece32k_Bert_text_sentence
## bert large configuration

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
           --micro-batch-size 8 \
           --vocab-file $VOCAB_FILE \
           --split 949,50,1 \
           --fp16"

OUTPUT_ARGS="--log-interval 10 \
             --save-interval 500 \
             --eval-interval 100 \
             --eval-iters 10 \
             --checkpoint-activations"

python pretrainWPbert.py \
       --tensor-model-parallel-size 1 \
       $BERT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH