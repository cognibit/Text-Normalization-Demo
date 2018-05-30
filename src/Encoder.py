"""
    Generate Required Encoding for XGBoost and DNC Model 
    

"""

import pickle
import numpy as np
from multiprocessing import Pool
import pandas as pd
import itertools

class XGBoostEncodingGenerator:
    
    def __init__(self,space_letter=0,max_num_features = 30,pad_size = 1,boundary_letter = -1):
        self.space_letter=space_letter
        self.max_num_features=max_num_features
        self.boundary_letter=boundary_letter
        self.pad_size=pad_size
        
    def context_window_transform(self,data, pad_size,flush_progress=True):
        pre = np.zeros(self.max_num_features)
        pre = [pre for x in np.arange(pad_size)]
        data = pre + data + pre
        neo_data = []
        for i in np.arange(len(data) - pad_size * 2):
            row = []
            if(flush_progress and i%100==0):
                print('Processed %f%%'%((i/(len(data) - pad_size * 2-1))*100),end='\r')
            for x in data[i : i + pad_size * 2 + 1]:
                row.append([self.boundary_letter])
                row.append(x)
            row.append([self.boundary_letter])
            merged=list(itertools.chain(*row))
            neo_data.append(merged)
        if(flush_progress):
            print('Processed 100%        ',end='\r')
        return neo_data
    
    def encode(self,df):
        x_data = []
        for x in df['before'].values:
            x_row = np.ones(self.max_num_features, dtype=int) * self.space_letter
            for xi, i in zip(list(str(x)), np.arange(self.max_num_features)):
                x_row[i] = ord(xi)
            x_data.append(x_row)
        return np.array(self.context_window_transform(x_data, self.pad_size), dtype = np.int16)
    
    def encode_csv(self,csv_file):
        csv=pd.read_csv(csv_file)
        encoding=self.encode(csv)
        print('Finished Encoding %s'%csv_file)
        return encoding
    
    def encode_csvs_parallel(self,csv_list,n_threads=8):
        """
        Encode Multiple CSVs in parallel 
        """
        if (n_threads < 1):
            assert ('nthreads is 1, cannot proceeed!')
        threads = Pool(n_threads)
        all_enc=threads.map(self.encode_csv,csv_list)
        return all_enc