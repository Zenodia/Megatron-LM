#INPUT_JSON_FILE=/workspace/SVdata/cleaned/SVcleaned.json
INPUT_JSON_FILE=/workspace/SVdata/raw/json/79803/SV_CC100Sprakbank.json
OUTPUT_PATH=/workspace/SVdata/ngc/SVCC100_sprakbank_32kWP
VOCAB_FILE=/workspace/SVdata/wp/SVCC100sprakbank_HFWP32k-vocab.txt
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

