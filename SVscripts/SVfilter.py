# -*- coding: utf-8 -*-
import sys
import multiprocessing
import os
from multiprocessing import Pool,cpu_count
import nltk
import nltk
sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
import argparse

def cut(sentence):
    sents=sent_tokenizer.tokenize(sentence)
    sents=[s for s in sents if len(s)>1]
    if len(sents)>=2: # we filter only more than 2 sentences in each doc
        return ''.join(sents)
    else:
        return ''

def main(args):
    num_cpus_to_use=int(args.num_cpu)
    lines=[]
    with open(args.input_dir, 'r', errors='ignore') as f:
        for k, line in enumerate(f, 1):
            line=line.replace('~','')
            line=line.replace('…','')
            line=line.strip()
            if not line:
                continue
            else:
                if len(line)>0 and line not in ['\n','','\r\n','\t',' ']:
                    lines.append(line)
                    k+=1
    print("Finish reading the file with number of non_empty lines ", str(k))
    # 创建进程池
    pool = Pool(num_cpus_to_use)
    data = pool.imap(cut,lines, 512)
    i=0
    outfile = open(args.output_dir, 'a')
    for i,dat in enumerate(data):
        if len(dat) >0:
            #print(i,dat,type(dat))
            outfile.write(dat+'\n')
        if i%1000000==0:
            print("processed {} million lines ...\n".format(str(i/1000000)))
            print("sample dat at i",i, dat)
    print("processing finished !")
    f.close()
    outfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default=None, type=str,  help='input file path')
    parser.add_argument('--output_dir', default=None, type=str, 
                        help='output file path')
    parser.add_argument('--num_cpu', default=48, type=int, 
                        help='number of cpus used for multiprocesing')
    args = parser.parse_args()
    main(args)




