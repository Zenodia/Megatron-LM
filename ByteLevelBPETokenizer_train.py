# ByteLevelBPETokenizer Represents a Byte-level BPE as introduced by OpenAI with their GPT-2 model
from transformers import GPT2TokenizerFast
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer, AutoModelWithLMHead
from tokenizers import ByteLevelBPETokenizer
from tokenizers import Tokenizer
import os
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', default=None, type=str,  help='path to the txt file with white space as separator')
    parser.add_argument('--bpe_path', default=None, type=str, help='output bpe path')
    parser.add_argument('--vocab_file', default=None, type=str, help='supply pretrained vocab.json file')
    parser.add_argument('--merge_file', default=None, type=str, help='supply pretrained merges.txt file')
    parser.add_argument('--vocab_size', default=None, type=int, 
                        help='specify the vocab_size when training HF BPE, one usually take vocab size divisible by 8 such as: 8k/16k/32k/48k/64k')
    parser.add_argument('--from_pretrained', action='store_true',
                       help='download pretrained gpt2 tokenizer from huiggingface transformer and then retrained on new dataset')
    parser.add_argument('--is_german', action='store_true',
                       help='download pretrained dbmdz/german-gpt2 tokenizer from huiggingface transformer and then retrained on new dataset')
    args = parser.parse_args()  
    if args.from_pretrained:
        if args.is_german:
            tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
        else:
            pretrained_weights = 'gpt2'        
            tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_weights)
            tokenizer.pad_token = tokenizer.eos_token
        print('---------- download one of the following vocab.json and merges.txt files ----------')
        print()
        print('vocab_files_names:',tokenizer.vocab_files_names)
        print()
        for k,v in tokenizer.pretrained_vocab_files_map.items():
            print(k)
        for kk,vv in v.items():
            print('- ',kk,':',vv)
        print()    
        print('vocab_size:',tokenizer.vocab_size)
        ByteLevelBPE_tokenizer_vocab_size = tokenizer.vocab_size # in case one wants to use the original vocab size from pretrained tokenizers
        #tokenizer_en.pre_tokenizer = Whitespace() # for non-European languages , such as pre-jiebacut-chinese 
        
        vocab_file=args.vocab_file
        merge_file=args.merge_file
        tokenizer=ByteLevelBPETokenizer.from_file(vocab_file, merge_file)
    else:

        tokenizer = ByteLevelBPETokenizer()
        # Get list of paths to corpus files
    paths = [args.infile]
    # Get GPT2 tokenizer_en vocab size
    
    if args.vocab_size is not None:
        ByteLevelBPE_tokenizer_vocab_size=args.vocab_size
        print("vocab size = ",  ByteLevelBPE_tokenizer_vocab_size)    

    # Customize training with <|endoftext|> special GPT2 token
    # ByteLevelBPETokenizer Represents a Byte-level BPE as introduced by OpenAI with their GPT-2 model
    # Customize training with <|endoftext|> special GPT2 token
    tokenizer.train(files=paths, 
                                    vocab_size=ByteLevelBPE_tokenizer_vocab_size, 
                                    min_frequency=2, 
                                    special_tokens=["<|endoftext|>"])

    # Get sequence length max of 1024
    tokenizer.enable_truncation(max_length=1024)
    # save tokenizer
    path_to_save_bpe_models = args.bpe_path
    if not os.path.exists(path_to_save_bpe_models):
        os.makedirs(path_to_save_bpe_models, exist_ok=True)
        
    print("saving trained model to : ",path_to_save_bpe_models)
    tokenizer.save_model(path_to_save_bpe_models)
    print("model saved ! \n\n\n")
    print("testing ...\n\n\n")
    test_txt="Rödluvan sprang ut och hämtade stora stenar, som de fyllde vargens mage med."
    output = tokenizer.encode(test_txt)
    for tok, i in zip(output.tokens, output.ids):
        print("token = {} , id ={}".format(str(tok), str(i)))