CHECKPOINT_PATH=./zh_ckpt
VOCAB_FILE=/workspace/cn/bpe/32k/vocab.json
MERGE_FILE=/workspace/cn/bpe/32k/merges.txt
DATA_PATH=/workspace/dataset/demo_gpt_sentence_text_sentence

GPT_ARGS="--num-layers 36 \
          --hidden-size 2048 \
          --num-attention-heads 32 \
          --seq-length 512 \
          --max-position-embeddings 512 \
          --micro-batch-size 32 \
          --global-batch-size 32 \
          --lr 0.00015 \
          --train-iters 500000 \
          --lr-decay-iters 320000 \
          --lr-decay-style cosine \
          --vocab-file $VOCAB_FILE \
          --merge-file $MERGE_FILE \
          --lr-warmup-fraction .01 \
          --fp16"

OUTPUT_ARGS="--log-interval 10 \
             --save-interval 500 \
             --eval-interval 100 \
             --eval-iters 10 \
             --checkpoint-activations"

python pretrain_gpt.py \
       $GPT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \