INPUT_JSON_FILE=/workspace/SVdata/cleaned/SVcleaned.json
OUTPUT_PATH=/workspace/SVdata/preprocessed/SV_WordPiece32k_Bert
VOCAB_FILE=/workspace/SVdata/wp/SV_HFWordPiece_vocab32k-vocab.txt
NUM_CPUS=1


python tools/preprocessSVdata.py \
       --input $INPUT_JSON_FILE \
       --output-prefix $OUTPUT_PATH \
       --json-keys text \
       --vocab-file $VOCAB_FILE \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences \
       --workers $NUM_CPUS

