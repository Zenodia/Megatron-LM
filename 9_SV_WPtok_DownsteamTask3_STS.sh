## make sure to modify the following 
# in tasks>glue>finetune.py > line 48 inside function model_provider()
#    def model_provider():
#        args = get_args()
#        print_rank_0('building classification model for {} ...'.format(args.task))
#        return Classification(num_classes=num_classes, num_tokentypes= 2 5) change num_tokentypes from 2--> 5 
TRAIN_DATA="/workspace/SVdata/SuperLimSE/sts/out/SV_STS_train_with_index.tsv"
VALID_DATA="/workspace/SVdata/SuperLimSE/sts/out/SV_STS_test_with_index.tsv /workspace/SVdata/SuperLimSE/sts/out/SV_STS_dev_with_index.tsv"
CHECKPOINT_PATH='/workspace/SVdata/sv_ckpt/downstream_ckpt/'
VOCAB_FILE='/workspace/SVdata/wp/SV_HFWordPiece_vocab32k-vocab.txt'


PRETRAINED_CHECKPOINT='/workspace/SVdata/sv_ckpt/pretrained/wp_ckpt/'

COMMON_TASK_ARGS="--num-layers 24 \
                  --hidden-size 1024 \
                  --num-attention-heads 16 \
                  --seq-length 512 \
                  --max-position-embeddings 512 \
                  --fp16 \
                  --vocab-file $VOCAB_FILE "

COMMON_TASK_ARGS_EXT="--train-data $TRAIN_DATA \
                      --valid-data $VALID_DATA \
                      --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
                      --checkpoint-activations \
                      --save-interval 10000 \
                      --save $CHECKPOINT_PATH \
                      --log-interval 100 \
                      --eval-interval 1000 \
                      --eval-iters 10 \
                      --weight-decay 1.0e-2"

python tasks/main.py \
       --task SV_STS \
       $COMMON_TASK_ARGS \
       $COMMON_TASK_ARGS_EXT \
       --tokenizer-type BertWordPieceLowerCase \
       --epochs 5 \
       --micro-batch-size 256 \
       --lr 5.0e-5 \
       --lr-warmup-fraction  0.065
