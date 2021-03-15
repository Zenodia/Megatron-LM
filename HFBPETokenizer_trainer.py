import os , sys
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', default=None, type=str,  help='path to the txt files pre-split using jieba')
    parser.add_argument('--bpe_path', default=None, type=str, help='output bpe path')
    parser.add_argument('--vocab_size', default=None, type=int, 
                        help='specify the vocab_size when training HF BPE for chinese usually 8k/16k/32k/48k/64k')
    args = parser.parse_args()    
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    vocab_size=int(args.vocab_size) # do NOT generate 16000, empty merges.txt , preprocess_data.py results in error since merges.txt is empty)
    print("vocab_size for this training is : ", str(vocab_size))
    trainer = BpeTrainer(vocab_size=vocab_size, show_progress=True,special_tokens = ['[CLS]','[BOS]','[EOS]','[SEP]','[PAD]','[MASK]','[UNK]','[EOD]'])
    # since tokenizers >=10.0 , it is now using tokenizer.train([list of txt files], trainer)
    #infiles=[os.path.join(args.infiles_path,f) for f in os.listdir(args.infiles_path) if f.endswith('.txt')]    
    #print("list of input txt  files for training :", infiles)
    #if len(infiles)>1:
    #   tokenizer.train( infiles,trainer )
    infile=args.infile
    print("infile is :", infile)
    tokenizer.train( [infile],trainer )    
    print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))
    # You will see the generated files in the output.
    print("saving trained BPE model to : ", args.bpe_path)
    tokenizer.model.save(args.bpe_path)
    print("model saved ! \n\n\n")
    print("testing ...\n\n\n")
    test_txt="达尔尼克乡达尔尼克乡是罗马尼亚的乡份，位于该国中部，由科瓦斯纳县负责管辖，处于布加勒斯特以北170公里，每年平均降雨量1,048毫米，海拔高度577米，2007年人口952。"
    output = tokenizer.encode(test_txt)
    print(output.tokens)