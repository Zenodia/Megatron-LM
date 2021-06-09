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

"""QQP dataset."""

from megatron import print_rank_0
from tasks.data_utils import clean_text
from .data import GLUEAbstractDataset
import pandas as pd


LABELS = [0, 1 ,2]


class SV_STSDataset(GLUEAbstractDataset):

    def __init__(self, name, datapaths, tokenizer, max_seq_length,
                 test_label=0):
        self.test_label = test_label
        super().__init__('SVSTS', name, datapaths,
                         tokenizer, max_seq_length)

    def process_samples_from_single_path(self, filename):
        """"Implement abstract method."""
        print_rank_0(' > Processing {} ...'.format(filename))

        samples = []
        total = 0
        first = True
        is_test = False
        df=pd.read_csv(filename,sep='\t')        
        if first:
            first = False
            if 'test' in filename or 'dev' in filename :
                is_test = True
                print_rank_0('   reading {}, {}, and {} columns and '
                             'setting labels to {}'.format(
                                 df.at[0,'index'], df.at[0,'sentence1'],
                                 df.at[0,'sentence2'], df.at[0,'label']))
            else:
                assert 'train' in filename
                print_rank_0('   reading {}, {}, and {} columns and '
                             'setting labels to {}'.format(
                                 df.at[0,'index'], df.at[0,'sentence1'],
                                 df.at[0,'sentence2'], df.at[0,'label']))
                            
        n=len(df)                                  
        for ind in range(n):            
            text_a = clean_text(df.at[ind,'sentence1'].strip())
            text_b = clean_text(df.at[ind,'sentence2'].strip()) 
            unique_id = int(df.at[ind,'index'])
            label = int(round(float(df.at[ind,'label'])))
            assert len(text_a) > 0
            assert len(text_b) > 0
            assert label in LABELS
            assert unique_id >= 0

            sample = {'text_a': text_a,
                      'text_b': text_b,
                      'label': LABELS[label],
                      'uid': unique_id}
            total += 1
            samples.append(sample)

            if total % 50000 == 0:
                print_rank_0('  > processed {} so far ...'.format(total))
        return samples
