from __future__ import print_function
import tensorflow as tf
#from tensorflow.contrib import rnn
import numpy as np
import os
import os.path

# Open the training data from the static directory, read in all of the data,
# and convert it to lowercase.
all_data = ""
with open('traindata.txt', 'r') as f:
	all_data = f.read()
all_data = all_data.lower()

# set() - returns a set object of all of the unique items in a list. In this
#   case, the list is a sequence of characters comprising the entirety of the
#   text contained in 'all_data'
# The list() keyword explicitly converts the set into a list. The locations of
# elements of the char_vocab can now be translated into one-hot encodings for
# training and testing.
char_vocab = list(set(all_data))

num_chars = len(char_vocab)
num_timesteps = 50
num_hidden_units = 200
learning_rate = 0.001
batch_size = 32 # Number of character sequences in a batch
batches_per_epoch = 500
max_epochs = 30
max_steps = max_epochs * batches_per_epoch
sample_step = 200

def char_to_vector(char):
	char = char.lower()
	vector = np.zeros(num_chars)
	vector[char_vocab.index(char)] = 1
	return vector

def vector_to_char(vector):
	max_value = max(vector)
	max_index = np.where(vector==max_value)
	#print("max index", max_index[0].tolist()[0])
	return char_vocab[max_index[0].tolist()[0]]

# Given the number of timesteps, batch size, and the batch sequence, return matrices
# of training and testing data. Note that you're extracting 'num_timesteps' number of
# characters from the training set, 'batch_size' number of sequential sets of characters,
# and each character will be broken down into a 'num_chars' length one-hot encoded
# vector.
# The order of these indices *matters* because the input to the LSTM requires input
# data in a certain format.
def get_next_batch(num_timesteps, batch_size, batch_number):
	start_index = batch_number*batch_size*num_timesteps

	input_matrix = np.zeros((batch_size, num_timesteps, num_chars))
	output_matrix = np.zeros((batch_size, num_timesteps, num_chars))

	for i in range(batch_size):
		index = i*num_timesteps + batch_number*batch_size
		input_substring = all_data[index:index + num_timesteps]
		output_substring = all_data[index+1:index+num_timesteps+1]

		'''
		if i == 1:
			print("samples of input and output")
			print("in:",input_substring)
			print("out:",output_substring)
		'''

		for j in range(len(input_substring)):
			input_matrix[i,j,:] = char_to_vector(input_substring[j])
			output_matrix[i,j,:] = char_to_vector(output_substring[j])

	return input_matrix, output_matrix

def string_to_tensor(input_string):
	input_matrix = np.zeros((1, len(input_string), num_chars))
	for j in range(len(input_string)):
		input_matrix[1,j,:] = char_to_vector(input_string[j])
	return input_matrix


# The maximum number of steps you go through is 
#   max_epochs * batch_size / num_training_samples


# Converting this to a class so that object properties can be taken advantage of.
class RNN(object):

	def __init__(self, learning_rate, session):
		self.session = session

		with tf.variable_scope('stuff'):
			self.weights = tf.Variable(tf.random_normal([num_hidden_units, num_chars]))
			self.biases = tf.Variable(tf.random_normal([num_chars]))

			# Declaring this as a placeholder means that it must be fed with data 
			# at execution time.
			# This means that its value has to be specified using the feed_dict = {}
			# argument inside of Session.run(), Tensor.eval(), or Operation.run()
			# The value that you feed it is going to be a numpy array.
			self.feed_x = tf.placeholder(dtype=tf.float32, shape=(None, None, num_chars))
			self.feed_y = tf.placeholder(dtype=tf.float32, shape=(None, None, num_chars))
			self.state_r = tf.placeholder(dtype=tf.float32, shape=(None, num_hidden_units))
			self.state_c = tf.placeholder(dtype=tf.float32, shape=(None, num_hidden_units))
			# This converts the placeholders for the states 'r' and 'c' into a tuple 
			# so that they can be passed to the dynamic_rnn. Note that in the case of
			# multiple LSTM layers you'd have N 'r' and 'c' states to pass; one 'r' and
			# 'c' for each layer of the LSTM network.
			self.rnn_tuple_state = tf.contrib.rnn.LSTMStateTuple(self.state_r, self.state_c)

			# The cell state and the hidden state are vectors with the same number
			# of components. This is usually coupled with an op that constructs the
			# actual network - rnn.static_rnn, dynamic_rnn, etc.
			self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden_units, forget_bias=1.0, state_is_tuple=True)

			self.lstm_zero_state = self.lstm_cell.zero_state(batch_size, dtype=tf.float32)
			print(type(self.lstm_zero_state), self.lstm_zero_state)
			#print(session.run(self.lstm_zero_state))

			# 'output' should have dimensions of [batch_size, num_unrolls, num_hidden_units]
			# Since the RNN was defined as being dynamic, the amount of layer unrolling
			# can change from batch to batch.
			self.outputs, self.last_lstm_state = tf.nn.dynamic_rnn(cell=self.lstm_cell, inputs=self.feed_x, initial_state=self.rnn_tuple_state, dtype=tf.float32)
			print(self.outputs)

			self.local_field = tf.matmul(tf.reshape(self.outputs, [-1, num_hidden_units]), self.weights) + self.biases

			# What we're ultimately reducing.
			# This is an operation on the tensorflow graph that contains the 'local_field' object
			# returned from the rnn() method.
			self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.local_field, labels=self.feed_y))

			# Convert the local fields generated by the output of the LSTM into a softmax output.
			self.softmax_output = tf.nn.softmax(self.local_field)

			# Declare the optimizer and the op that you want to minimize.
			self.train_operation = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

	def train(self, batch_x, batch_y):
		# Need to add state_c and need to cut the zero state down... maybe?
		print("lstm zero state", self.lstm_zero_state[0])
		self.session.run(self.train_operation, feed_dict={self.feed_x:batch_x, self.feed_y:batch_y, self.state_r:self.lstm_zero_state[0].eval(), self.state_c:self.lstm_zero_state[1].eval()})

	def run(self, x, num_steps=25, delimiter=None):
		'''
		This method will create 'num_steps' unrollings of the network after passing
		the input tensor, 'x', to the network.
		'''
		test_output = ""

		lstm_out, lstm_state_out = self.session.run([self.outputs, self.last_lstm_state], feed_dict={self.feed_x:x, self.state_r:self.lstm_zero_state.eval(), self.state_c:self.lstm_zero_state.eval()})

		for j in range(lstm_out.shape[0]):
			test_output += vector_to_char(lstm_out[j])

		for i in range(num_steps):
			lstm_in = lstm_out
			lstm_state_in = lstm_state_out
			lstm_out, lstm_state_out = self.session.run([self.outputs, self.last_lstm_state], feed_dict={self.feed_x:lstm_in, self.state_r:lstm_state_in})
			for j in range(lstm_out.shape[0]):
				test_output += vector_to_char(lstm_out[j])

		return test_output


	
#local_field, final_state = rnn(feed_x, weights['out'], biases['out'])


# Declare a variable to contain the global variable initialization op.
init = tf.global_variables_initializer()

# Declare a saver for the ops on the graph.


with tf.Session() as sess:
	network = RNN(learning_rate, sess)
	saver = tf.train.Saver()
	# Initialize all of the variables we've declared to their default values.
	# In this context, you initialize the weights and biases to random
	# normal values.
	sess.run(init)

	#if os.path.isfile('checkpoint'):
	if False:
		saver.restore(sess, os.getcwd() + "\\lstmsmall")

		print(network.run(string_to_tensor('random stuff')))

	else:
		# Recall that one epoch is a forward pass and backward pass for every 
		# training sample.
		step = 1

		print("maximum number of steps:", max_steps)
		
		# The total number of steps is the number of batches that have been passed
		# through for training. So for one epoch there are
		#   num_training_samples/batch_size
		# steps.
		while step < max_steps:
			# The training batch has to have specific dimensions:
			#  (batch_size, num_unrolls, num_inputs)
			# num_unrolls - number of characters in the sequence
			# num_inputs - number of 'items' in the training vocabulary (28)
			# Since we're building a language model, the training_batch is also the
			# desired outputs, except the desired outputs are shifted by one word.
			train_batch_inputs, desired_batch_outputs = get_next_batch(num_timesteps, batch_size, step % batches_per_epoch)

			# Note that the elements of the feed dictionary are the placeholders
			# defined earlier in the program.
			network.train(train_batch_inputs, desired_batch_outputs)

			if step % sample_step == 0:
				print("-----------------------------")
				print("current step", step)
				#training_output = sess.run(softmax_output, feed_dict={feed_x: train_batch_inputs})
				training_output = network.run(train_batch_inputs, num_steps=0)

				#print(training_output.shape)
				# Just need to collect the max outputs of the softmax layer and
				# convert the indices back to characters. Hopefully that shit looks
				# decent.
				phrase = ""
				for i in range(training_output.shape[0]):
					phrase += vector_to_char(training_output[i])

				print(phrase)

			step += 1

		saver.save(sess, os.getcwd() + "\\lstmsmall")