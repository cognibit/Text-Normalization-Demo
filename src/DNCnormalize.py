"""Text Normalization using Differentiable Neural Computer"""

import numpy as np
import tensorflow as tf
from collections import OrderedDict

from lib.seq2seq import Seq2SeqModel


# ----------------------
# Model Flag Parameters
# ----------------------

FLAGS = tf.flags.FLAGS

# Network parameters
tf.flags.DEFINE_string('cell_type', 'dnc', 'RNN cell for encoder and decoder, default: lstm')
tf.flags.DEFINE_string('attention_type', 'bahdanau', 'Attention mechanism: (bahdanau, luong), default: bahdanau')
tf.flags.DEFINE_integer('hidden_units', 1024, 'Number of hidden units in each layer')
tf.flags.DEFINE_integer('depth', 1, 'Number of layers in each encoder and decoder')
tf.flags.DEFINE_integer('embedding_size', 20, 'Embedding dimensions of encoder and decoder inputs')
tf.flags.DEFINE_integer('num_decoder_symbols', 1500, 'Target vocabulary size')

tf.flags.DEFINE_boolean('use_residual', False, 'Use residual connection between layers')
tf.flags.DEFINE_boolean('attn_input_feeding', False, 'Use input feeding method in attentional decoder')
tf.flags.DEFINE_boolean('use_dropout', False, 'Use dropout in each rnn cell')
tf.flags.DEFINE_float('dropout_rate', 0.3, 'Dropout probability for input/output/state units (0.0: no dropout)')

tf.flags.DEFINE_integer('start_token', 0, 'Start Token')

# Training parameters
tf.flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')
tf.flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
tf.flags.DEFINE_integer('batch_size', 200, 'Batch size')
tf.flags.DEFINE_integer('max_load_batches', 20, 'Maximum # of batches to load at one time')
tf.flags.DEFINE_integer('display_freq', 100, 'Display training status every this iteration')
tf.flags.DEFINE_integer('save_freq', 11500, 'Save model checkpoint every this iteration')
tf.flags.DEFINE_integer('valid_freq', 1150000, 'Evaluate model every this iteration: valid_data needed')
tf.flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop)')
tf.flags.DEFINE_string('summary_dir', '../models/summary', 'Path to save model summary')
tf.flags.DEFINE_boolean('shuffle_each_epoch', True, 'Shuffle training dataset for each epoch')
tf.flags.DEFINE_boolean('sort_by_length', True, 'Sort pre-fetched minibatches by their target sequence lengths')
tf.flags.DEFINE_boolean('use_fp16', False, 'Use half precision float16 instead of float32 as dtype')

# Runtime parameters
tf.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
tf.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

#DNC Specific Parameters
tf.flags.DEFINE_integer('memory_size', 256 , 'DNC Memory Size')
tf.flags.DEFINE_integer('word_size', 64, 'DNC Memory Word Size')
tf.flags.DEFINE_integer('num_reads', 4, 'Num Read Heads')
tf.flags.DEFINE_integer('num_writes',1, 'Num Write Heads')
tf.flags.DEFINE_integer('clip_value',20, 'clips controller and core output values to between')

#Decode Parameters
tf.flags.DEFINE_integer('beam_width', 0, 'Beam Search Decoder')
tf.flags.DEFINE_integer('max_decode_step', 50, 'Number of steps to decode')


def normalize(enc_data, enc_len, batch_size=200):
	"""Normalize encoded data using the trained DNC model given"""	

	# Initiate TF session
	tf.reset_default_graph()
	dnc_predictions=[]

	with tf.Session() as sess:
		print('Using DNC model at {}'.format(FLAGS.model_dir))
		model=create_model_decode(flags=FLAGS, batch_size=batch_size)
		restore_model(model,sess)

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

def create_model_decode(flags, batch_size):
	config = dict(OrderedDict(sorted(flags.__flags.items())))
	model = Seq2SeqModel(config,'decode',batch_size)
	return model

def restore_model(model, sess):
	ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)

	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print('Reloading model parameters...')
		model.restore(sess, ckpt.model_checkpoint_path)
	else:
		if not os.path.exists(self.FLAGS.model_dir):
			os.makedirs(self.FLAGS.model_dir)
		print('Created new model parameters..')
		
	return None