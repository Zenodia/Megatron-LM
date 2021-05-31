# modify pretrain_bert.py 
# duplicate the original pretrain_bert.py and change the name to Zeno_modified_pretrainBPEbert.py
# line 41 change num_tokentypes 2-->5
# line 44 change num_tokentypes 2-->5
# line 49 change num_tokentypes 2-->5
# line 52 change num_tokentypes 2-->5
# line 155 change tokenizer_type BertWordPieceLowerCase--> HFBPETokenizer

CHECKPOINT_PATH=./zh_ckpt/
VOCAB_FILE=./SVdata/bpe/32k/vocab.json
MERGE_FILE=./SVdata/bpe/32k/merges.txt

DATA_PATH=./SVdata/preprocessed/SV32k_Bert__text_sentence
## bert large configuration

BERT_ARGS="--num-layers 24 \
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
           --fp16"

OUTPUT_ARGS="--log-interval 10 \
             --save-interval 500 \
             --eval-interval 100 \
             --eval-iters 10 \
             --checkpoint-activations"


python pretrainBPEbert.py \
       --tensor-model-parallel-size 1 \
       $BERT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH
