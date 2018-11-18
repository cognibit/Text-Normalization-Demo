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
Text Normalization using Differentiable Neural Computer

"""

import os
import numpy as np
import tensorflow as tf
from collections import OrderedDict

from lib.seq2seq import Seq2SeqModel


# ----------------------
# Model Flag Parameters
# ----------------------
config={
    'cell_type':'dnc',
    'attention_type':'bahdanau',
    'hidden_units':1024,
    'depth':1,
    'embedding_size':32,
    'memory_size':256,
    'word_size':64,
    'num_writes':1,
    'num_reads':5,
    'clip_value':20,
    'beam_width':1,
    'max_decode_step':150,
    'use_residual':False,
    'attn_input_feeding':True,
    'use_dropout':False,
    'dropout_rate':0.3,
    'use_fp16':False
    
}


def normalize(enc_data, enc_len, model_path,batch_size=200,use_memory=True):
	"""Normalize encoded data using the trained DNC model given"""	

	# Initiate TF session
	tf.reset_default_graph()
	dnc_predictions=[]
	with tf.Session() as sess:
		print('Using DNC model at {}'.format(model_path))
		model=create_model_decode(batch_size=batch_size,use_memory=use_memory)
		restore_model(model,sess,model_path)

		num_batches=int(enc_data.shape[0]/batch_size)
		print('Number of batches: {}'.format(num_batches))

		for i in range(num_batches):
			predict=model.predict(sess,enc_data[i*batch_size:i*batch_size+batch_size],
								   enc_len[i*batch_size:i*batch_size+batch_size])
			predict = np.split(predict,batch_size,axis=0)
			dnc_predictions.extend(predict)

			if i%(int(num_batches/25)) == 0:
				print('Normalized {} out of {}'.format((i+1)*batch_size,
												num_batches*batch_size))

		#Process the last batch by adding zeros to the end
		if(enc_data.shape[0]%batch_size != 0):
			lastbatch = enc_data[num_batches*batch_size:]
			lastbatch_len= enc_len[num_batches*batch_size:]
			lastbatch=np.concatenate((lastbatch,np.zeros([batch_size-lastbatch.shape[0],
												lastbatch.shape[1]])),axis=0)
			
			lastbatch_len=np.concatenate((lastbatch_len,
				np.ones([batch_size-lastbatch_len.shape[0]])),axis=0)
			
			predict=model.predict(sess,lastbatch,lastbatch_len)
			predict=np.split(predict,batch_size,axis=0)
			dnc_predictions.extend(predict)

	return dnc_predictions

def create_model_decode(batch_size,use_memory):
	model = Seq2SeqModel(config,'decode',batch_size,use_memory=use_memory)
	return model

def restore_model(model, sess, model_path):
	print('Reloading model parameters...')
	model.restore(sess, model_path)
	return None