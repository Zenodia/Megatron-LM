CHECKPOINT_PATH=./zh_ckpt/
VOCAB_FILE=./SVdata/bpe/32k/vocab.json
MERGE_FILE=./SVdata/bpe/32k/merges.txt

DATA_PATH=./SVdata/preprocessed/SV32k_Bert__text_sentence
             
python pretrainBPEbert.py \
        --pipeline-model-parallel-size 1 \
        --num-layers 24 \
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
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE\
        --split 949,50,1 \
        --fp16 \
        --log-interval 100000 \
        --save-interval 100000 \
        --eval-interval 100000 \
        --eval-iters 100 \
        --checkpoint-activations \ 
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH