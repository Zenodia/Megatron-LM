INPUT_JSON_FILE=/workspace/cn/sample_1mil.json
OUTPUT_PATH=/workspace/dataset/demo_text_sentence
VOCAB_FILE=/workspace/cn/bpe/32k/vocab.json
MERGE_FILE=/workspace/cn/bpe/32k/merges.txt
NUM_CPUS=32

python tools/zeno_zh_process.py \
       --input $INPUT_JSON_FILE \
       --output-prefix $OUTPUT_PATH \
       --json-keys text \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --dataset-impl mmap \
       --tokenizer-type HFBPETokenizer  \
       --split-sentences \
       --sentence-splitter snowNLP \
       --tensor-model-parallel-size 1 \
       --workers $NUM_CPUS