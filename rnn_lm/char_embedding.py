from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import os
from data_handler import DataHandler

class AutoEncoder(object):
	def __init__(self, num_nodes, input_size, session, scope_name='encoder'):
		self.session = session

		with tf.variable_scope(scope_name):
			self.feed_x = tf.placeholder(shape=(None, None), dtype=tf.float32)

			self.in_weights = tf.Variable(tf.random_normal(shape=\
				(input_size, num_nodes), stddev=0.01), name='encw')
			self.in_biases = tf.Variable(tf.random_normal(shape=\
				(num_nodes,), stddev=0.01), name='encb')

			self.out_weights = tf.Variable(tf.random_normal(shape=\
				(num_nodes, input_size), stddev=0.01))
			self.out_biases = tf.Variable(tf.random_normal(shape=\
				(input_size,), stddev=0.01))

			self.encoder = tf.tanh(tf.matmul(self.feed_x, self.in_weights) + \
				self.in_biases)

			self.local_fields = tf.matmul(self.encoder, self.out_weights) + \
				self.out_biases

			self.output = tf.nn.softmax(self.local_fields)

			#self.cost = tf.reduce_mean(tf.squared_difference(self.output,
			#	self.feed_x))
			self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
				logits=self.local_fields, labels=self.feed_x))

			self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)\
				.minimize(self.cost)

	def train(self, batch_x):
		feeds = {self.feed_x:batch_x}
		fetches = [self.cost, self.optimizer]

		cost, _ = self.session.run(fetches, feeds)

		return cost

	def run(self, test_x):
		feeds = {self.feed_x:[test_x]}
		fetches = self.output

		result = self.session.run(fetches, feeds)

		return result


def main(argv):
	'''
	Make the embedding happen on the 128 characters of the ASCII table using
	a one-hot encoding of each characters.
	'''

	ascii_data = np.zeros((128, 128))
	for i in range(128):
		ascii_data[i,i] = 1.0

	num_iterations = 0
	max_iterations = 8000

	sess = tf.Session()
	encoder = AutoEncoder(25, ascii_data.shape[0], sess)
	sess.run(tf.global_variables_initializer())

	while num_iterations < max_iterations:
		cost = encoder.train(ascii_data)

		num_iterations += 1

		if num_iterations % 500 == 0:
			print('-----------------------------------')
			print("cost:", cost)
			start = np.random.choice(range(128))
			#print("start:", start)
			print("num iterations:", num_iterations)
			for i in range(start, start+5):
				result = encoder.run(ascii_data[i])
				print("desired:", np.where(ascii_data[i]>0)[0][0])
				print("actual:", np.where(result[0]==max(result[0]))[0][0])
				#print("result:", result[0])
				
			print('\n\n')

	train_vars = tf.trainable_variables()
	save_vars = [i for i in train_vars if '/enc' in i.name]
	#save_vars_dict = {i.name.split(':')[0][-1]:sess.run(i) for i in save_vars}
	for i in save_vars:
		print(i.name)

	#print(save_vars_dict)

	enc_params = sess.run(save_vars)
	print("enc_params:", enc_params)
	np.savez('charembedding', enc_params[0], enc_params[1])
	#np.savez('charembedding', save_vars_dict)

	params = np.load('charembedding.npz')
	for i in params.files:
		print(i)
		print(params[i].shape)

if __name__ == '__main__':
	main(sys.argv[1:])