# Copyright 2018 Cognibit Solutions LLP.
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
# ==============================================================================
"""

Utility Modules for Text Normalization
"""

import pickle
import numpy as np
from multiprocessing import Pool


class Encoder:
    def __init__(self, vocab_file, wlook=3, time_major=False):
        self.vocab_file = vocab_file
        self.wlook = wlook
        self.time_major = time_major

    def encode(self, df, nthreads=8):
        if (nthreads < 1):
            assert ('nthreads is 1, cannot proceeed!')
        else:
            row_len = df.shape[0]
            batch_len = int(row_len / nthreads)
            last_batch = row_len % nthreads
            batches = []
            for i in range(nthreads):
                if (i != nthreads - 1):
                    batches.append(df.iloc[i * batch_len:i * batch_len + batch_len])
                else:
                    batches.append(df.iloc[i * batch_len:])
            threads = Pool(nthreads)
            encoded_dfs = threads.map(self.run_single_batch, batches)
            encoding, encoding_len = zip(*encoded_dfs)
            col_len = 0
            for e in encoding:
                if (e.shape[1] > col_len):
                    col_len = e.shape[1]
            encoding = list(encoding)
            for i in range(len(encoding)):
                encoding[i] = np.concatenate((encoding[i], np.zeros([encoding[i].shape[0]
                                                                        , col_len - encoding[i].shape[1]])), axis=1)
            encoding = np.concatenate(encoding)
            encoding_len = np.concatenate(encoding_len)
            return encoding, encoding_len

    def run_single_batch(self, df):
        batch_gen = EncodingGenerator(self.vocab_file, self.wlook, self.time_major)
        return batch_gen.encode(df)

        
class EncodingGenerator:
    def __init__(self, vocab_file, wlook=3, time_major=False):
        self.train_grp = None
        self.row_len = None
        with open(vocab_file, 'rb') as handle:
            self.vocab_dict = pickle.loads(handle.read())
        self.sent_id = 0
        self.token_id = 0
        self.row_count = 0
        self.wlook = wlook
        self.time_major = time_major
        self.group_keys = None

    def __input_lookup(self, char):
        if (char in self.vocab_dict['input']):
            return self.vocab_dict['input'][char]
        else:
            return self.vocab_dict['input']['<UNK>']

    def __input_word_lookup(self, word):
        lookups = []
        word = str(word)
        print
        for c in word:
            lookups.append(self.__input_lookup(c))
        return lookups

    def __next_element(self):
        sent = self.train_grp.get_group(self.group_keys[self.sent_id])
        if (self.token_id > sent.shape[0] - 1):
            self.sent_id = (self.sent_id + 1) % self.train_grp.ngroups
            self.token_id = 0
            sent = self.train_grp.get_group(self.group_keys[self.sent_id])
        token_count = sent.shape[0]
        row_dict = dict()
        new_row = []
        for k in range(-self.wlook, self.wlook + 1):
            if (k == 0):
                new_row.append(self.__input_lookup('<norm>'))
                lookup = self.__input_word_lookup(sent.iloc[k + self.token_id, :]['before'])
                new_row.extend(lookup)
                new_row.append(self.__input_lookup('</norm>'))
                new_row.append(self.__input_lookup(' '))
            elif ((self.token_id + k < 0 or self.token_id + k > token_count - 1) == False):
                lookup = self.__input_word_lookup(sent.iloc[k + self.token_id, :]['before'])
                new_row.extend(lookup)
                new_row.append(self.__input_lookup(' '))
        new_row.append(self.__input_lookup('<EOS>'))
        self.token_id = self.token_id + 1
        return new_row

    def encode(self, df):
        self.train_grp = df.groupby(by='sentence_id')
        self.row_len = df.shape[0]
        self.group_keys = list(self.train_grp.groups.keys())
        input_batches = []
        max_inp_len = 0
        for b in range(self.row_len):
            i = self.__next_element()
            input_batches.append(i)
            if (len(i) > max_inp_len):
                max_inp_len = len(i)
        # Add the padding characters
        input_batches_len = np.zeros([self.row_len])
        count = 0
        for b in input_batches:
            input_batches_len[count] = len(b)
            count = count + 1
            for i in range(0, max_inp_len - len(b)):
                b.append(self.__input_lookup('<PAD>'))

        input_batches = np.array(input_batches)

        if (self.time_major == True):
            input_batches = input_batches.T

        return input_batches, input_batches_len


class Normalized2String:
    def __init__(self, vocab_file):
        with open(vocab_file, 'rb') as handle:
            self.vocab_dict = pickle.loads(handle.read())
            output_id_dict = self.vocab_dict['output']
            self.output_id_dict_rev = {v: k for k, v in output_id_dict.items()}

    def to_str(self, prediction):
        """
        prediction : A 1D numpy array
        """
        final_str = ''
        for id in prediction:
            word = self.__output_lookup_inverse(id)
            if word == '<EOS>':
                break
            else:
                final_str = final_str +' '+ str(word)
        return final_str[1:]

    def __output_lookup_inverse(self, id):
        return self.output_id_dict_rev[id]