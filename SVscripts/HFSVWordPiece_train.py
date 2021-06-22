from tokenizers import BertWordPieceTokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import  WordPieceTrainer
import argparse
import os, sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', default=None, type=str,  help='path to the txt files pre-split using jieba')
    parser.add_argument('--out_path', default=None, type=str, help='output bpe path')
    parser.add_argument('--vocab_size', default=None, type=int, 
                        help='specify the vocab_size when training HF BPE for chinese usually 8k/16k/32k/48k/64k')
    args = parser.parse_args()    
    special_tokens=['[CLS]','[BOS]','[EOS]','[SEP]','[PAD]','[MASK]','[UNK]','[EOD]']
    infile=args.infile
    out_file_path= args.out_path
    # Initialize an empty tokenizer
    tokenizer = BertWordPieceTokenizer(clean_text=True, handle_chinese_chars=False, strip_accents=False, lowercase=True)
    # use a custom delimiter 
    tokenizer.pre_tokenizer = Whitespace()
    vocab_size=args.vocab_size
    print("vocab size: {} , {}".format(type(vocab_size),vocab_size))
    trainer = WordPieceTrainer(vocab_size=vocab_size, show_progress=True,special_tokens = special_tokens,continuing_subword_prefix='##')
    tokenizer.train([infile],vocab_size=vocab_size, min_frequency=2,limit_alphabet=1000000, special_tokens=special_tokens, show_progress=True, wordpieces_prefix='##')
    print("saving trained WordPiece model to : ", args.out_path)
    print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))
    # Save the files
    tokenizer.save_model( out_file_path, 'SVCC100sprakbank_HFWP32k')
