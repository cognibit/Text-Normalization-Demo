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
This script prepares the raw data for the next stage of normalization.
"""

import os
import sys
import pandas as pd
from multiprocessing import Pool


def preprocessing(file):
    print('Launch Processing of {}'.format(file))
    output = file+'_processed.csv'
 
    # By default, Pandas treats double quote as enclosing an entry so it includes all tabs and newlines in that entry
    # until it reaches the next quote. To escape it we need to have the quoting argument set to QUOTE_NONE or 3 as
    # given in the documentation - [https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html]
    raw_data = pd.read_csv(file, header=None, sep='\t', quoting = 3, names=['semiotic', 'before', 'after'])

    # Generating sentence and word token ids
    # Our text normalization approach requires sentence and token ids to encode and generate batches
    data = pd.DataFrame(columns=['sentence_id',
                                 'token_id',
                                 'semiotic',
                                 'before',
                                 'after'])
    # initialize columns and iterator
    sentence_id = 0
    token_id = -1

    # heavy processing ahead
    for row in raw_data.itertuples():
        # look for end of sentences
        if row.semiotic == '<eos>' and row.before == '<eos>':
            sentence_id += 1
            token_id = -1
            continue
        else:
            token_id += 1
            
        new_row = {'sentence_id': sentence_id,
                   'token_id': token_id,
                   'semiotic': row.semiotic,
                   'before': row.before,
                   'after': row.after}
        data = data.append(new_row, ignore_index=True)    
        print('Processing Sentence#{} of {}'.format(sentence_id, file))

    # **Transforming 'after' tokens**  
    # From the above mentioned paper:
    # ```
    # Semiotic class instances are verbalized as sequences
    # of fully spelled words, most ordinary words are left alone (rep-
    # resented here as <self>), and punctuation symbols are mostly
    # transduced to sil (for “silence”).
    # ```
    # Hence we transform as follows:
    # 1. sil is replaced with < self >
    # 2. < self > is replaced with the before column
    # 
    sil_mask = (data['after'] == 'sil')
    data.loc[sil_mask, 'after'] = '<self>' 
    self_mask = (data['after'] == '<self>')
    data.loc[self_mask, ('after')] = data.loc[self_mask, 'before']

    # Exporting Data
    data.to_csv(output, index=False)
    print('Done {}'.format(file))
    return True

def split_dataframe(df, size=10*1024*1024):
    """Splits huge dataframes(CSVs) into smaller segments of given size in bytes"""
    
    # size of each row
    row_size = df.memory_usage().sum() / len(df)
    # maximum number of rows in each segment
    row_limit = int(size // row_size)
    # number of segments
    seg_num = (len(df)+row_limit-1)//row_limit
    # split df into segments
    segments = [df.iloc[i*row_limit : (i+1)*row_limit] for i in range(seg_num)]

    return segments


if __name__ == '__main__':
    path = sys.argv[1]
    jobs = int(sys.argv[2])

    # split large CSVs
    for dirpath, _, filenames in os.walk(path):
        for file in filenames:
            df = pd.read_csv(os.path.join(dirpath, file),header=None, sep='\t', quoting = 3, names=['semiotic', 'before', 'after'])            
            df_splits = split_dataframe(df, 10*1024*1024)
            # save each split and delete original
            for i in range(len(df_splits)):
                split_file = file+'_part{}'.format(i+1)
                df_splits[i].to_csv(os.path.join(dirpath, split_file))
            os.remove(os.path.join(dirpath, file)) 
    print("Splitted original file into chunks...")

    files=[]
    for dirpath, _, filenames in os.walk(path):
        for file in filenames:
            files.append(os.path.join(dirpath, file))

    pool=Pool(jobs)
    pool.map(preprocessing, files)

