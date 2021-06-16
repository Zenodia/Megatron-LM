# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MNLI dataset."""

from megatron import print_rank_0
from tasks.data_utils import clean_text
from .data import SVGLUEAbstractDataset
import pandas as pd


LABELS = {'-1': 0, '0': 1 , '1' :2 }


class SVSentimentDataset(SVGLUEAbstractDataset):

    def __init__(self, name, datapaths, tokenizer, max_seq_length):        
        super().__init__('SVSentimentDataset', name, datapaths,
                         tokenizer, max_seq_length)

    def process_samples_from_single_path(self, filename):
        """"Implement abstract method."""
        print_rank_0(' > Processing {} ...'.format(filename))

        samples = []
        total = 0
        first = True
        is_test = False
        df=pd.read_csv(filename,sep='\t')
        if 'test' in filename or 'dev' in filename :
            is_test = True
            print_rank_0(
                '   reading index {}, and text {} columns and setting '
                'labels to {}'.format(
                    df.at[0,'index'], df.at[0,'text'], df.at[0,'label_simple']))
        else:
            print_rank_0(
                '   reading index {}, and text {} columns and setting '
                'labels to {}'.format(
                    df.at[0,'index'], df.at[0,'text'], df.at[0,'label_simple']))
        
        n=len(df)                                  
        for ind in range(n):            
            text = clean_text(df.at[ind,'text'].strip())                
            unique_id = int(df.at[ind,'index'])
            #label = str(round(float(df.at[ind,'label_simple'])))
            label = str(df.at[ind,'label_simple'])
            #print(i, unique_id, label , text)
            assert len(text) > 0
            assert label in LABELS
            assert unique_id >= 0

            sample = {'text': text,
                      'label': LABELS[label],
                      'uid': unique_id}
            total += 1
            samples.append(sample)

            if total % 50000 == 0:
                print_rank_0('  > processed {} so far ...'.format(total))

        print_rank_0(' >> processed {} samples.'.format(len(samples)))
        return samples
