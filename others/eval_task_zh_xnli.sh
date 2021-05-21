## make sure to modify the following 
# in tasks>glue>finetune.py > line 48 inside function model_provider()
#    def model_provider():
#        args = get_args()
#        print_rank_0('building classification model for {} ...'.format(args.task))
#        return Classification(num_classes=num_classes, num_tokentypes= 2 5) change num_tokentypes from 2--> 5 
TRAIN_DATA="/workspace/dataset/zh_glue/xnli/train_3labels.tsv"
VALID_DATA="/workspace/dataset/zh_glue/xnli/test.tsv /workspace/dataset/zh_glue/xnli/dev.tsv"
CHECKPOINT_PATH='/workspace/dataset/zh_glue/finetune_ckpt/32k/latest/'
#VOCAB_FILE='/workspace/cn/bpe/32k/vocab.json'
#MERGE_FILE='/workspace/cn/bpe/32k/merges.txt'
#remember to change the checkpoint to either 800000 or 1800000 for performance test 
#PRETRAINED_CHECKPOINT='/workspace/dataset/zh_glue/bpe_ckpt/32k/'
VOCAB_FILE='/workspace/cn/bpe/32k/vocab.json'
MERGE_FILE='/workspace/cn/bpe/32k/merges.txt'
PRETRAINED_CHECKPOINT='/workspace/dataset/zh_glue/bpe_ckpt/32k/1838654/' # iter_3680000

COMMON_TASK_ARGS="--num-layers 24 \
                  --hidden-size 1024 \
                  --num-attention-heads 16 \
                  --seq-length 128 \
                  --max-position-embeddings 512 \
                  --fp16 \
                  --vocab-file $VOCAB_FILE \
                  --merge-file $MERGE_FILE"

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
       --task ZH_XNLI \
       $COMMON_TASK_ARGS \
       $COMMON_TASK_ARGS_EXT \
       --tokenizer-type HFBPETokenizer \
       --epochs 3 \
       --batch-size 256 \
       --lr 5.0e-5 \
       --warmup 0.065