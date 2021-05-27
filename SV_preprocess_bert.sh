INPUT_JSON_FILE=/workspace/SVdata/cleaned/SVcleaned.json
OUTPUT_PATH=/workspace/SVdata/test/SV_WordPiece32k_Bert
VOCAB_FILE=/workspace/SVdata/bpe/32k/vocab.json
MERGE_FILE=/workspace/SVdata/bpe/32k/merges.txt
NUM_CPUS=1


python tools/preprocessSVdata.py \
       --input $INPUT_JSON_FILE \
       --output-prefix $OUTPUT_PATH \
       --json-keys text \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --dataset-impl mmap \
       --tokenizer-type HFBPETokenizer \
       --split-sentences \
       --workers $NUM_CPUS

