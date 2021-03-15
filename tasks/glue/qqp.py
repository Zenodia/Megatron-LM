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
import jieba

LABELS = [0, 1]


class QQPDataset(GLUEAbstractDataset):

    def __init__(self, name, datapaths, tokenizer, max_seq_length,
                 test_label=0):
        self.test_label = test_label
        super().__init__('QQP', name, datapaths,
                         tokenizer, max_seq_length)

    def process_samples_from_single_path(self, filename):
        """"Implement abstract method."""
        print_rank_0(' > Processing {} ...'.format(filename))

        samples = []
        total = 0
        first = True
        is_test = False
        with open(filename, 'r') as f:
            uid=0
            print(" ================== filename is {} : ================".format(filename))
            for line in f:
                row = line.strip().split('\t')                
                if 'dev' in filename or 'text' in filename :
                    is_test = True                    
                    print_rank_0('   reading {}, {}, and {} columns and '
                                     'setting labels to {}'.format(
                                         uid, row[1].strip(),
                                         row[2].strip(), self.test_label))
                    uid+=1
                else:                        
                    assert len(row)==3
                    print_rank_0('    reading {}, {}, {}, and {} columns'
                                    ' ...'.format(
                                        uid, row[0].strip(),
                                        row[1].strip(), row[2].strip()))
                    uid+=1
                

                if len(row)==3:               
                    uid = int(uid)
                    text_a = clean_text(row[0].strip())
                    text_b = clean_text(row[1].strip())
                    label = int(row[2].strip()) #self.test_label
                    assert len(text_a) > 0
                    assert len(text_b) > 0
                    if len(text_a) == 0 :
                        print_rank_0('***WARNING*** zero length a, '
                                     'skipping: {}'.format(row))
                        continue
                    if len(text_b) == 0 :
                        print_rank_0('***WARNING*** zero length b, '
                                     'skipping: {}'.format(row))
                        continue

                else:
                    print_rank_0('***WARNING*** index error, '
                                     'skipping: {}'.format(row))
                    continue
                    
                assert label in LABELS
                assert uid >= 0
                text_a = ' '.join(list(jieba.cut(text_a)))
                text_b = ' '.join(list(jieba.cut(text_b)))
                sample = {'uid': uid,
                          'text_a': text_a,
                          'text_b': text_b,
                          'label': label}
                total += 1
                samples.append(sample)

                if total % 50000 == 0:
                    print_rank_0('  > processed {} so far ...'.format(total))

        print_rank_0(' >> processed {} samples.'.format(len(samples)))
        return samples
