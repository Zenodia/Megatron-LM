import json
import os, sys
import numpy as np
import nltk
from sb_corpus_reader import SBCorpusReader
import random
out_path='/workspace/SVdata/raw/out/'
done_xml=['familjeliv-allmanna-ekonomi.xml','gp2013.xml']
xml_files=os.listdir('raw')
print(xml_files)
for xml_f in xml_files:
    if xml_f.endswith('.xml'):        
        fname='_'.join(xml_f.split('.')[0].split('-'))+'.txt'  
        print("fname: ",fname)

def write2csv(out_path, fname, sents):
    f=open(out_path+fname,'a')
    for s in sents:
        if len(s)>=13:
            s_text=' '.join(s)
            f.write(s_text+'\n')    
    f.close()
    print("finish processing ",fname)

for xml_f in xml_files:
    if xml_f.endswith('.xml') and xml_f not in done_xml:
        try:            
            corpus = SBCorpusReader('./raw/'+xml_f)
            print("begining processing : ", xml_f)
            sents=corpus.sents()
            print(sents[:2])
            #n=len(sents)
            #rn=random.randint(0,n-1)
            #print("a random sample of sentence : \n".format(' '.join(sents[rn])))
            fname='_'.join(xml_f.split('.')[0].split('-'))+'.txt'  
            print("write to : ",fname)
            write2csv(out_path,fname,sents)
            print('-----'*10)
        except :
            continue
