from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import sys
import thread_ops


# Open the training data from the static directory, read in all of the data,
# and convert it to lowercase.
all_data = ""
with open('traindata.txt', 'r') as f:
	all_data = f.read()
all_data = all_data.lower()

# set() - returns a set object of all of the unique items in a list. In this
#   case, the list is a sequence of characters comprising the entirety of the
#   text contained in 'all_data'.
# The list() keyword explicitly converts the set into a list. The locations of
# elements of the char_vocab can now be translated into one-hot encodings for
# training and testing.
char_vocab = list(set(all_data))

# num_chars - Number of unique characters used in the training set.
# num_timesteps - Number of unrollings to do for training data. (Is this used?)
# num_hidden_units - Number of hidden neurons in the LSTM cell.
# learning_rate - Well...
# batch_size - The number of training data vectors to feed to the network at
#   a time.
# batches_per_epoch - Number of batches that will be considered one epoch's 
#   worth of training.
# max_epochs - Total number of training loops to be executed.
# max_steps - Total number of batch passes to the network.
# sample_step_percentage - Sample output from training will be displayed at
#   every integer multiple of this percentage during the training steps
num_chars = len(char_vocab)
num_timesteps = 50
num_hidden_units = 200
num_lstm_layers = 2
learning_rate = 0.001
batch_size = 32 
batches_per_epoch = 500
max_epochs = 30
max_steps = max_epochs * batches_per_epoch
sample_step_percentage = .01

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

def string_to_tensor(string_):
	'''
	Converts a string into a tensor (numpy array). This is intended for running the
	network on a sequence of characters.
	The output will have the shape
	[1 string_length vocabulary_size]
	'''
	tensor = np.zeros((1, len(string_), num_chars))
	for j in range(len(string_)):
		tensor[0,j,:] = char_to_vector(string_[j])
	return tensor

def get_next_batch(num_timesteps, batch_size, batch_number):
	'''
	Given the number of timesteps, batch size, and the batch sequence, return matrices
	of training and testing data. Note that you're extracting 'num_timesteps' number of
	characters from the training set, 'batch_size' number of sequential sets of characters,
	and each character will be broken down into a 'num_chars' length one-hot encoded
	vector.
	The order of these indices *matters* because the input to the LSTM requires input
	data in a certain format.
	The next modification of this method should be allowing for random sampling from the
	data set (without replacement).
	'''
	start_index = batch_number*batch_size*num_timesteps

	input_matrix = np.zeros((batch_size, num_timesteps, num_chars))
	output_matrix = np.zeros((batch_size, num_timesteps, num_chars))

	for i in range(batch_size):
		index = i*num_timesteps + batch_number*batch_size
		input_substring = all_data[index:index + num_timesteps]
		output_substring = all_data[index+1:index+num_timesteps+1]

		for j in range(len(input_substring)):
			input_matrix[i,j,:] = char_to_vector(input_substring[j])
			output_matrix[i,j,:] = char_to_vector(output_substring[j])

	return input_matrix, output_matrix

def get_random_batch(num_timesteps, batch_size):
	start_index = int((len(all_data) - batch_size - 1)*np.random.random_sample())

	input_matrix = np.zeros((batch_size, num_timesteps, num_chars))
	output_matrix = np.zeros((batch_size, num_timesteps, num_chars))

	for i in range(batch_size):
		index = int((len(all_data) - batch_size - 1)*np.random.random_sample())
		input_substring = all_data[index:index + num_timesteps]
		output_substring = all_data[index+1:index+num_timesteps+1]

		for j in range(len(input_substring)):
			input_matrix[i,j,:] = char_to_vector(input_substring[j])
			output_matrix[i,j,:] = char_to_vector(output_substring[j])

	return input_matrix, output_matrix


# Converted this process from a regular method to a class so that object properties
# can be taken advantage of.
class RNN(object):

	def __init__(self, learning_rate, session, scope_name):
		#####################################################################
		#
		# Started modifying the network to accept multiple layers. Use a
		# list to store the number of units at a given layer. You'll have
		# to make the following things into lists:
		#   - zero states (for initializing the network)
		#   - the basic lstm cells (these will also have to be passed to the
		#     dynamic rnn module)
		# Need to modify the feed dictionaries for training/testing.
		# Might want to segment the various layers into separate scopes.
		#
		#####################################################################
		self.session = session

		with tf.variable_scope(scope_name):
			self.weights = tf.Variable(tf.random_normal([num_hidden_units, num_chars], stddev=0.01))
			self.biases = tf.Variable(tf.random_normal([num_chars], stddev=0.01))

			# Declaring this as a placeholder means that it must be fed with data 
			# at execution time.
			# This means that its value has to be specified using the feed_dict = {}
			# argument inside of Session.run(), Tensor.eval(), or Operation.run()
			# The value that you feed it is going to be a numpy array.
			self.feed_x = tf.placeholder(dtype=tf.float32, shape=(None, None, num_chars))
			self.feed_y = tf.placeholder(dtype=tf.float32, shape=(None, None, num_chars))
			#self.state_r = tf.placeholder(dtype=tf.float32, shape=(None, num_hidden_units))
			#self.state_c = tf.placeholder(dtype=tf.float32, shape=(None, num_hidden_units))
			self.states = [[tf.placeholder(dtype=tf.float32, shape=(None, num_hidden_units))]*2 for i in range(num_lstm_layers)]
			# This converts the placeholders for the states 'r' and 'c' into a tuple 
			# so that they can be passed to the dynamic_rnn. Note that in the case of
			# multiple LSTM layers you'd have N 'r' and 'c' states to pass; one 'r' and
			# 'c' for each layer of the LSTM network.
			#self.rnn_tuple_states = tf.contrib.rnn.LSTMStateTuple(self.state_r, self.state_c)
			self.rnn_tuple_states = tuple([tf.contrib.rnn.LSTMStateTuple(state[0], state[1]) for state in self.states])

			# The cell state and the hidden state are vectors with the same number
			# of components. This is usually coupled with an op that constructs the
			# actual network - rnn.static_rnn, dynamic_rnn, etc.
			
			self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden_units, forget_bias=1.0, state_is_tuple=True)
			self.multi_lstm = tf.contrib.rnn.MultiRNNCell([self.lstm_cell]*num_lstm_layers, state_is_tuple=True)

			# Not necessary to generate the zero states.
			# Maybe you can declare them here if you end up using them a lot.
			self.lstm_zero_states = self.multi_lstm.zero_state(batch_size, dtype=tf.float32)
			#print(type(self.lstm_zero_state), self.lstm_zero_state)
			#print(session.run(self.lstm_zero_state))

			# 'output' should have dimensions of [batch_size, num_unrolls, num_hidden_units]
			# Since the RNN was defined as being dynamic, the amount of layer unrolling
			# can change from batch to batch.
			# 
			self.outputs, self.last_lstm_state = tf.nn.dynamic_rnn(cell=self.multi_lstm, inputs=self.feed_x, initial_state=self.rnn_tuple_states, dtype=tf.float32)
			print("output from dynamic rnn", self.outputs)

			# The local field of the softmax output (argument of the softmax).
			# This should return a matrix.
			# There's a bias for each output node, and the outputs of the dynamic_rnn are for
			# each timestep, so the output *should* be a matrix of outputs at each timestep.
			self.local_field = tf.matmul(tf.reshape(self.outputs, [-1, num_hidden_units]), self.weights) + self.biases
			print("size of local field", self.local_field)

			# What we're ultimately reducing.
			# This is an operation on the tensorflow graph that contains the 'local_field' object
			# returned from the rnn() method.
			self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.local_field, labels=self.feed_y))

			# Convert the local fields generated by the output of the LSTM into a softmax output.
			self.softmax_output = tf.nn.softmax(self.local_field)

			# Declare the optimizer and the op that you want to minimize.
			self.train_operation = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

	def train(self, batch_x, batch_y):
		
		zero_states = self.session.run(self.lstm_zero_states)
		feed_dict = {}
		feed_dict[self.feed_x] = batch_x
		#print("size of batch_x:", batch_x.shape)
		#for i in range(len(zero_states)):
		#	print("size of zero states:", zero_states[i][0].shape, zero_states[i][1].shape)

		feed_dict[self.feed_y] = batch_y
		for i in range(len(self.states)):
			feed_dict[self.states[i][0]] = zero_states[i][0]
			feed_dict[self.states[i][1]] = zero_states[i][1]
		self.session.run(self.train_operation, feed_dict=feed_dict)

	def run(self, x, num_steps=25, delimiter=None):
		'''
		This method will create 'num_steps' unrollings of the network after passing
		the input tensor, 'x', to the network.
		'''
		test_output = ""
		#zero_state_r = self.session.run(self.lstm_zero_state[0])
		#zero_state_c = self.session.run(self.lstm_zero_state[1])
		#zero_state_r = np.zeros((x.shape[0], num_hidden_units))
		#zero_state_c = np.zeros((x.shape[0], num_hidden_units))
		zero_states = self.session.run(self.multi_lstm.zero_state(x.shape[0], dtype=tf.float32))
		#feed_dict={self.feed_x:x, self.state_r:zero_state_r, self.state_c:zero_state_c}
		feed_dict={}
		feed_dict[self.feed_x] = x
		#for i in range(len(zero_states)):
		#	print("size of zero states:", zero_states[i][0].shape, zero_states[i][1].shape)
		#print("size of input:", x.shape)
		for i in range(len(self.states)):
			feed_dict[self.states[i][0]] = zero_states[i][0]
			feed_dict[self.states[i][1]] = zero_states[i][1]
		softmax_out, _, lstm_state_out = self.session.run([self.softmax_output, self.outputs, self.last_lstm_state], feed_dict=feed_dict)
		print("softmax_out", softmax_out)

		for j in range(softmax_out.shape[0]):
			test_output += vector_to_char(softmax_out[j])

		for i in range(num_steps):
			lstm_in = np.zeros(softmax_out.shape)
			lstm_in[np.where(softmax_out==np.amax(softmax_out))] = 1.0
			print("softmax out:", softmax_out)
			#lstm_state_r_in = lstm_state_out[0]
			#lstm_state_c_in = lstm_state_out[1]
			lstm_state_in = lstm_state_out
			#feed_dict={self.feed_x:[lstm_in], self.state_r:lstm_state_r_in, self.state_c:lstm_state_c_in}
			feed_dict = {}
			feed_dict[self.feed_x] = [lstm_in]
			for i in range(len(self.states)):
				feed_dict[self.states[i][0]] = lstm_state_in[i][0]
				feed_dict[self.states[i][1]] = lstm_state_in[i][1]
			softmax_out, lstm_out, lstm_state_out = self.session.run([self.softmax_output, self.outputs, self.last_lstm_state], feed_dict=feed_dict)
			for j in range(lstm_out.shape[0]):
				test_output += vector_to_char(softmax_out[j])

		return test_output

def main(argv):
	load_checkpoint = False
	if len(argv) > 0:
		load_checkpoint = True
		checkpoint_file = argv[0]
	# Start tf session. This is passed to the RNN class.
	sess = tf.Session()

	cud = os.getcwd()
	weight_saver = thread_ops.weightThread()

	network = RNN(learning_rate, sess, 'stuff')

	# Initialize all of the variables inside of the 'sess' tensorflow session.
	sess.run(tf.global_variables_initializer())

	# Declare a saver for the ops on the graph.
	saver = tf.train.Saver()

	if load_checkpoint and os.path.isfile(os.path.join(cud, checkpoint_file)):
	#if False:
		saver.restore(sess, os.path.join(cud, "lstmsmall"))

		print(network.run(string_to_tensor('random stuff')))

	else:
		# Recall that one epoch is a forward pass and backward pass for all 
		# training data.
		step = 1

		#print("maximum number of steps:", max_steps)
		# Retrieve the list of trainable variables. The goal is to save these
		# weights as a grayscale PNG to show how the values evolve over time.
		trainable_vars = tf.trainable_variables()
		
		# The total number of steps is the number of batches that have been passed
		# through for training. So for one epoch there are
		#   num_training_samples/batch_size
		# steps.
		while step < max_steps:
			# The training batch has to have specific dimensions:
			#   (batch_size, num_unrolls, vocab_length)
			# num_unrolls - number of characters in the training sequence
			# vocab_length - number of unique items in the training vocabulary
			# Since we're building a language model, the training_batch is also the
			# desired outputs, except the desired outputs are shifted by one word.
			#train_batch_inputs, desired_batch_outputs = get_next_batch(num_timesteps, batch_size, step % batches_per_epoch)
			train_batch_inputs, desired_batch_outputs = get_random_batch(num_timesteps, batch_size)

			# Note that the elements of the feed dictionary are the placeholders
			# defined earlier in the program.
			network.train(train_batch_inputs, desired_batch_outputs)
			print(step)
			if step % int(sample_step_percentage*float(max_steps)) == 0:
				print("-----------------------------------------------------")
				print("current step", step)

				training_output = network.run(train_batch_inputs, num_steps=0)
				print("training output:\n", training_output)
				#print("local field:\n", sess.run(network.local_field))

				#weights = sess.run(trainable_vars)
				#weight_saver.run(weights)

			step += 1

		saver.save(sess, os.path.join(cud, "lstmsmall"))

if __name__ == "__main__":
	main(sys.argv[1:])