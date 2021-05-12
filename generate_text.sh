CHECKPOINT_PATH=/workspace/gpt_ckpt/
VOCAB_FILE=/workspace/gpt_ckpt/vocab.json
MERGE_FILE=/workspace/gpt_ckpt/merges.txt

GPT_ARGS="--num-layers 24 \
          --hidden-size 1024 \
          --num-attention-heads 16 \
          --seq-length 1024 \
          --max-position-embeddings 1024 \
          --micro-batch-size 4 \
          --global-batch-size 8 \
          --lr 0.00015 \
          --train-iters 500000 \
          --lr-decay-iters 320000 \
          --lr-decay-style cosine \
          --vocab-file $VOCAB_FILE \
          --merge-file $MERGE_FILE \
          --lr-warmup-fraction .01 \
          --fp16"

MAX_OUTPUT_SEQUENCE_LENGTH=512
TEMPERATURE=1.0
TOP_P=0.9
NUMBER_OF_SAMPLES=0
OUTPUT_FILE=samples.json

python tools/generate_samples_gpt.py \
       $GPT_ARGS \
       --load $CHECKPOINT_PATH \
       --out-seq-length $MAX_OUTPUT_SEQUENCE_LENGTH \
       --temperature $TEMPERATURE \
       --genfile $OUTPUT_FILE \
       --num-samples $NUMBER_OF_SAMPLES \
       --top_p $TOP_P \
       --recompute