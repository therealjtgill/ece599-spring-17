from __future__ import print_function
from datetime import datetime
import tensorflow as tf
import numpy as np
import os
import sys
import thread_ops
from data_handler import *
from utils import *

class GRULM(object):

	def __init__(self, session, scope_name, vocab_size=128,
		num_rnn_layers=2, num_rnn_units=256):

		self.session = session
		self.num_rnn_units = num_rnn_units
		self.num_rnn_layers = num_rnn_layers
		self.vocab_size = vocab_size
		self.scope_name = scope_name

		dt = tf.float32
		char_embedding = np.load('charembedding.npz')

		with tf.variable_scope(self.scope_name):

			self.feed_in = tf.placeholder(dtype=dt,
				shape=(None, None, vocab_size))
			self.feed_out = tf.placeholder(dtype=dt,
				shape=(None, None, vocab_size))
			self.feed_learning_rate = tf.placeholder(dtype=dt,
				shape=())

			batch_size = tf.shape(self.feed_in)[0]
			seq_length = tf.shape(self.feed_in)[1]
			lr = self.feed_learning_rate

			embedding_weights = tf.constant(char_embedding['arr_0'])
			embedding_biases = tf.constant(char_embedding['arr_1'])

			#in_weights = tf.get_variable(name='cew', trainable=False,
			#	initializer=embedding_weights)
			#in_biases = tf.get_variable(name='ceb', trainable=False,
			#	initializer=embedding_biases)
			#embedding_weights = tf.Variable(char_embedding['arr_0'])
			#embedding_biases = tf.Variable(char_embedding['arr_1'])

			out_weights = tf.Variable(tf.random_normal(
				(num_rnn_units, vocab_size), stddev=0.01))
			out_biases = tf.Variable(tf.random_normal(
				(vocab_size,), stddev=0.01))

			rnn_cell = [tf.contrib.rnn.GRUCell(num_units=num_rnn_units) \
				for _ in range(num_rnn_layers)]

			rnn_state_size = [cell.state_size for cell in rnn_cell]

			self.hidden_states = [tf.placeholder(dtype=dt,
				shape=(None, s)) for s in rnn_state_size]

			self.multi_rnn = tf.contrib.rnn.MultiRNNCell(rnn_cell)

			num_tiles = batch_size
			expanded_weights = tf.expand_dims(embedding_weights, axis=0)
			tiled_weights = tf.tile(expanded_weights, [num_tiles, 1, 1])

			embedding_layer = tf.tanh(
				tf.matmul(self.feed_in, tiled_weights) + embedding_biases)

			rnn_out_raw, self.rnn_last_state = tf.nn.dynamic_rnn(
				cell=self.multi_rnn, inputs=embedding_layer,
				initial_state=tuple(self.hidden_states), dtype=dt)

			rnn_out = tf.tanh(rnn_out_raw)
			rnn_out_flat = tf.reshape(rnn_out, [-1, num_rnn_units])

			logits_flat = tf.matmul(rnn_out_flat, out_weights) + out_biases
			logits = tf.reshape(logits_flat,
				[batch_size, seq_length, vocab_size])
			self.softmax_out = tf.nn.softmax(logits)

			feed_out_flat = tf.reshape(self.feed_out, [-1, vocab_size])
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
				logits=logits, labels=feed_out_flat))

			optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)

			grads_and_vars = optimizer.compute_gradients(self.loss)
			capped_grads = [(tf.clip_by_value(grad, -10., 10.), var) \
				for grad, var in grads_and_vars]

			self.train_op = optimizer.apply_gradients(capped_grads)

	def train(self, batch_in, batch_out, lr=0.0001):

		batch_size = batch_in.shape[0]
		seq_length = batch_in.shape[1]
		num_rnn_layers = self.num_rnn_layers
		dt = tf.float32

		zero_states = self.session.run(self.multi_rnn.zero_state(batch_size, dt))

		feeds = {
			self.feed_in:batch_in,
			self.feed_out:batch_out,
			self.feed_learning_rate:lr,
		}

		for i in range(num_rnn_layers):
			feeds[self.hidden_states[i]] = zero_states[i]

		fetches = [
			self.loss,
			self.train_op
		]

		loss, _ = self.session.run(fetches, feed_dict=feeds)

		return loss

	def validate(self, valid_in, valid_out):

		batch_size = valid_in.shape[0]
		seq_length = valid_in.shape[1]
		num_rnn_layers = self.num_rnn_layers
		dt = tf.float32
		#num_rnn_units = self.num_rnn_units
		#zero_states = np.zeros((num_rnn_layers, 1, num_hidden_units))
		zero_states = self.session.run(self.multi_rnn.zero_state(batch_size, dt))

		feeds = {
			self.feed_in:valid_in,
			self.feed_out:valid_out,
		}
		for i in range(num_rnn_layers):
			feeds[self.hidden_states[i]] = zero_states[i]

		fetches = self.loss

		loss = self.session.run(fetches, feed_dict=feeds)
		perplexity = np.exp(loss)
		return perplexity

	def run(self, test_in, num_steps=25, delimiter=None):

		batch_size = 1
		seq_length = test_in.shape[0]
		num_rnn_layers = self.num_rnn_layers
		dt = tf.float32

		zero_states = self.session.run(self.multi_rnn.zero_state(batch_size, dt))
		rnn_next_state = zero_states

		probs = []
		feeds = {self.feed_in:[test_in]}
		fetches = [self.softmax_out, self.rnn_last_state]

		if len(test_in) < 1:
			raise Exception('test_in input to lm.run() must contain text')

		for j in range(num_rnn_layers):
			feeds[self.hidden_states[j]] = rnn_next_state[j]

		prob, rnn_next_state = self.session.run(fetches, feed_dict=feeds)
		
		for i in range(num_steps):
			one_hot = soft_prob_to_one_hot(prob[0][-1])
			probs.append(one_hot)
			
			feeds[self.feed_in] = [[one_hot]]

			for j in range(num_rnn_layers):
				feeds[self.hidden_states[j]] = rnn_next_state[j]

			prob, rnn_next_state = self.session.run(fetches, feed_dict=feeds)

		return probs

	def save(self, filename, save_dir=''):
		save_path = os.path.join(save_dir, filename)
		self.saver.save(self.session, save_path + '.ckpt')

	def set_saver(self, saver):
		self.saver = saver

def decrease_lr(loss, threshold, factor, lr):
	if len(loss) <= 1:
		rate = lr

	else:
		dp = (loss[-2] - loss[-1])/loss[-2]
		if dp < threshold:
			rate = lr * factor
		else:
			rate = lr
	return rate


def cross_train_lm(halving_threshold, sequence_length, sample_step, 
	max_halvings, max_steps, batch_size, corpora, model, save_dir):
	
	T = sequence_length
	S = sample_step
	M = max_steps
	H = max_halvings
	B = batch_size

	for corpus in corpora:
		step = 1
		num_halvings = 0
		learning_rate = 0.1

		set_ppl = []
		while num_halvings < H and step < M:
			train_in, train_out = corpus.get_random_batch(T, B, 'validation')
			loss = model.train(train_in, train_out, learning_rate)
			#print('cost:', cost)
			if ((step % S) == 0) and (step != 0):
				valid_sets = [(c.get_random_batch(T, B, 'validation')) \
					for c in corpora]

				test_in, test_out = corpus.get_random_batch(T, B, 'validation')
				set_ppl.append(model.validate(test_in, test_out))

				ppl = [model.validate(d[0], d[1]) for d in valid_sets]
				ppl_str = ', '.join([str(p) for p in ppl])
				save_text(ppl_str, save_dir, 'perplexity')

				new_lr = decrease_lr(set_ppl, halving_threshold,
					0.5, learning_rate)
				if new_lr != learning_rate:
					num_halvings += 1
					learning_rate = new_lr
					print('\tlearning rate decreased:', learning_rate)
				print('\nstep: ', step, 'loss:', loss)
				print('corpus:', corpus.name)
				for i in range(len(valid_sets)):
					print('\t', corpora[i].name, ' perplexity:', ppl[i])
				print('\tlearning rate halvings:', num_halvings)
				seed = string_to_tensor('this morning')
				print(probs_to_string(model.run(seed, num_steps=200)))
			step += 1

		model.save(corpus.name, save_dir)

def main(argv):
	batch_size = 192
	halving_threshold = 0.003
	sequence_length = 30
	sample_step = 200
	max_halvings = 15
	max_steps = 10000

	date = datetime.now()
	date = str(date).replace(' ', '').replace(':', '-')
	
	data_dir = make_dir('data')
	save_dir = make_dir(date)

	shakespeare = DataHandler('shakespeare.train.txt',
		'shakespeare.test.txt', 'shakespeare.valid.txt', data_dir)
	shakespeare.name = 'shakespeare'
	penntreebank = DataHandler('ptb.train.txt',
		'ptb.test.txt', 'ptb.valid.txt', data_dir)
	penntreebank.name = 'penntreebank'

	corpora = (shakespeare, penntreebank)

	session = tf.Session()
	lm = GRULM(session, 'basicrnn', DataHandler.vocab_size)
	lm.set_saver(tf.train.Saver())
	session.run(tf.global_variables_initializer())

	cross_train_lm(halving_threshold, sequence_length, sample_step, 
		max_halvings, max_steps, batch_size, corpora, lm, save_dir)

if __name__ == '__main__':
	main(sys.argv[1:])