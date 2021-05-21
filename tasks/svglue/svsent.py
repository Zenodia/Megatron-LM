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


LABELS = {'1': 0, '2': 1 , '3' :2 ,'4':3 ,'5':4}


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
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                row = line.strip().split('\t')
                if i==0 :
                    first = False
                    if 'test' in filename or 'dev' in filename :
                        is_test = True
                        print_rank_0(
                            '   reading index {}, and text {} columns and setting '
                            'labels to {}'.format(
                                row[0].strip(), row[-2].strip(), row[-1].strip()))
                    else:
                        print_rank_0('    reading index {} , text {}, and  label {} columns '
                                     '...'.format(
                                         row[0].strip(), row[-2].strip(),row[-1].strip()))                    
                else:
                    text = clean_text(row[-2].strip())                
                    unique_id = i
                    label = str(round(float(row[-1].strip())))
                    #print(i , label , text)
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
