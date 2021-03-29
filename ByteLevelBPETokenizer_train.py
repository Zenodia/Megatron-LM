# ByteLevelBPETokenizer Represents a Byte-level BPE as introduced by OpenAI with their GPT-2 model
from transformers import GPT2TokenizerFast
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import ByteLevelBPETokenizer
import os
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', default=None, type=str,  help='path to the txt file with white space as separator')
    parser.add_argument('--bpe_path', default=None, type=str, help='output bpe path')
    parser.add_argument('--vocab_size', default=None, type=int, 
                        help='specify the vocab_size when training HF BPE, one usually take vocab size divisible by 8 such as: 8k/16k/32k/48k/64k')
    parser.add_argument('--from_pretrained', action='store_true',
                       help='download pretrained gpt2 tokenizer from huiggingface transformer and then retrained on new dataset')
    args = parser.parse_args()  
    if args.from_pretrained:
        pretrained_weights = 'gpt2'
        tokenizer_en = GPT2TokenizerFast.from_pretrained(pretrained_weights)
        tokenizer_en.pad_token = tokenizer_en.eos_token
        ByteLevelBPE_tokenizer_pt_vocab_size = tokenizer_en.vocab_size
        #tokenizer_en.pre_tokenizer = Whitespace()

    ByteLevelBPE_tokenizer_pt = ByteLevelBPETokenizer()
    # Get list of paths to corpus files
    paths = ['/workspace/cn/raw_wiki/zh_wiki_jiebacut.txt']
    # Get GPT2 tokenizer_en vocab size
    
    if args.vocab_size is not None:
        ByteLevelBPE_tokenizer_pt_vocab_size=args.vocab_size
        print("vocab size = ",  ByteLevelBPE_tokenizer_pt_vocab_size)    

    # Customize training with <|endoftext|> special GPT2 token
    # ByteLevelBPETokenizer Represents a Byte-level BPE as introduced by OpenAI with their GPT-2 model
    # Customize training with <|endoftext|> special GPT2 token
    ByteLevelBPE_tokenizer_pt.train(files=paths, 
                                    vocab_size=ByteLevelBPE_tokenizer_pt_vocab_size, 
                                    min_frequency=2, 
                                    special_tokens=["<|endoftext|>"])

    # Get sequence length max of 1024
    ByteLevelBPE_tokenizer_pt.enable_truncation(max_length=1024)
    # save tokenizer
    ByteLevelBPE_tokenizer_pt_rep = '/workspace/cn/bpe_vocab_demo/exp/'
    print("saving trained model to : ",ByteLevelBPE_tokenizer_pt_rep)
    ByteLevelBPE_tokenizer_pt.save_model(ByteLevelBPE_tokenizer_pt_rep)
    print("model saved ! \n\n\n")
    print("testing ...\n\n\n")
    test_txt="达尔尼克乡达尔尼克乡是罗马尼亚的乡份，位于该国中部，由科瓦斯纳县负责管辖，处于布加勒斯特以北170公里，每年平均降雨量1,048毫米，海拔高度577米，2007年人口952。"
    output = ByteLevelBPE_tokenizer_pt.encode(test_txt)
    for tok, i in zip(output.tokens, output.ids):
        print("token = {} , id ={}".format(str(tok), str(i)))