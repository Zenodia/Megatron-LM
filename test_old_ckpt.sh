# modify pretrain_bert.py 
# duplicate the original pretrain_bert.py and change the name to Zeno_modified_pretrainBPEbert.py
# line 41 change num_tokentypes 2-->5
# line 44 change num_tokentypes 2-->5
# line 49 change num_tokentypes 2-->5
# line 52 change num_tokentypes 2-->5
# line 155 change tokenizer_type BertWordPieceLowerCase--> HFBPETokenizer

# adding this flag < --override-lr-scheduler > to overwirte lr scheduler and migrating the ckpt to new megatron 

vocab_size='32k'
CHECKPOINT_PATH=./dataset/old_ckpt/
VOCAB_FILE=./cn/bpe/32k/vocab.json
MERGE_FILE=./cn/bpe/32k/merges.txt

DATA_PATH=./dataset/old_ckpt/all6ZH_text_sentence

BERT_ARGS="--num-layers 24 \
           --hidden-size 1024 \
           --num-attention-heads 16 \
           --seq-length 512 \
           --max-position-embeddings 512 \
           --override-lr-scheduler \
           --lr 0.0001 \
           --lr-decay-style linear \
           --min-lr 1.0e-5 \
           --weight-decay 1e-2 \
           --clip-grad 1.0 \
           --train-iters 2160000 \
           --min-lr 1.0e-5 \
           --lr-decay-iters 990000 \
           --lr-warmup-fraction .01 \
           --micro-batch-size 16 \
           --vocab-file $VOCAB_FILE \
           --merge-file $MERGE_FILE\
           --split 949,50,1 \
           --fp16"



OUTPUT_ARGS="--log-interval 10 \
             --save-interval 500 \
             --eval-interval 100 \
             --eval-iters 10 \
             --checkpoint-activations"

python Zeno_modifed_pretrainBPEbert.py \
       --tensor-model-parallel-size 1 \
       $BERT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH
