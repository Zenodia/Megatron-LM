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
from .data import GLUEAbstractDataset

LABELS = {'contradictory': 0, 'entailment': 1, 'neutral': 2}


class SV_XNLIDataset(GLUEAbstractDataset):

    def __init__(self, name, datapaths, tokenizer, max_seq_length,
                 test_label='contradictory'):
        self.test_label = test_label
        super().__init__('SV_XNLI', name, datapaths,
                         tokenizer, max_seq_length)
    def get_data_point(self,row):
        text_a = clean_text(row[1].strip())
        text_b = clean_text(row[2].strip())
        unique_id = int(row[0].strip())
        label = row[-1].strip()       
        assert len(text_a) > 0
        assert len(text_b) > 0
        assert label in LABELS
        assert unique_id >= 0
        text_a = text_a
        text_b = text_b        
        sample = {'text_a': text_a,
                  'text_b': text_b,
                  'label': LABELS[label],
                  'uid': unique_id}
        return sample
    def process_samples_from_single_path(self, filename):
        """"Implement abstract method."""
        print_rank_0(' > Processing {} ...'.format(filename))
        samples = []
        total = 0
        first = True
        is_test = False
        with open(filename, 'r') as f:
            print_once=0
            for line in f:
                row = line.strip().split('\t')
                if len(row)<=3001 :
                    first = False
                    is_test = True
                    if print_once==0:
                        print_rank_0(
                            '  testing dataset reading {}, {} and {} columns and setting '
                            'labels to {}'.format(
                            row[1].strip(), row[2].strip(),
                            row[-3].strip(), self.test_label))
                    else:
                        sample=self.get_data_point(row)                        
                        total += 1
                        samples.append(sample)
                    print_once+=1
                else:
                    is_test = False
                    if print_once==0:
                        print_rank_0(' training dataset reading {} , {}, {}, and {} columns '
                                    '...'.format(
                                    row[1].strip(), row[2].strip(),
                                    row[0].strip(), row[-1].strip()))                    
                    if print_once>0:                        
                        sample=self.get_data_point(row)                        
                        total += 1
                        samples.append(sample)
                    print_once+=1  
                if total % 1000 == 0:
                    print_rank_0('  > processed {} so far ...'.format(total))
        print_rank_0(' >> processed {} samples.'.format(len(samples)))
        return samples
