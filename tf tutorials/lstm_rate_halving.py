from __future__ import print_function
from datetime import datetime
import tensorflow as tf
import numpy as np
import os
import sys
import thread_ops

# Open the training data from the static directory, read in all of the data,
# and convert it to lowercase.
train_data = ""
with open('traindata.txt', 'r') as f:
	train_data = f.read()
train_data = train_data.lower()

valid_data = ""
with open('validationdata.txt', 'r') as f:
	valid_data = f.read()
valid_data = valid_data.lower()

test_data = ""
with open('testdata.txt', 'r') as f:
	test_data = f.read()
test_data = test_data.lower()

# set() - returns a set object of all of the unique items in a list. In this
#   case, the list is a sequence of characters comprising the entirety of the
#   text contained in 'train_data'.
# Adding all of the data together to ensure that all possible characters are
# accounted for.
char_vocab = list(set(train_data + valid_data + test_data))

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
num_timesteps = 100
num_hidden_units = 256
num_lstm_layers = 2
batch_size = 64
max_steps = 10000
sample_step_percentage = .01
sample_weight_percentage = 0.005

def char_to_vector(char):
	char = char.lower()
	vector = np.zeros(num_chars)
	vector[char_vocab.index(char)] = 1.0
	return vector

def vector_to_char(vector):
	#print("shape of vector in vector_to_char:", vector.shape)
	#max_value = max(vector)
	index = np.random.choice(range(num_chars), p=vector)
	#print("vector being converted:", vector)
	#print("current vocab:", char_vocab)
	#print("index selected:", sampled_index)
	#print("max index", max_index[0].tolist()[0])
	#return char_vocab[max_index[0].tolist()[0]]
	return char_vocab[index]

def string_to_tensor(string_):
	'''
	Converts a string into a tensor (numpy array). This is intended for running the
	network on a sequence of characters.
	The output will have the shape
	[1, string_length, vocabulary_size]
	'''
	tensor = np.zeros((len(string_), num_chars))
	for j in range(len(string_)):
		tensor[j,:] = char_to_vector(string_[j])
	return tensor

def get_random_batch(num_timesteps, batch_size, data):

	input_matrix = np.zeros((batch_size, num_timesteps, num_chars))
	output_matrix = np.zeros((batch_size, num_timesteps, num_chars))

	for i in range(batch_size):
		index = int((len(data) - num_timesteps - 1)*np.random.rand())
		input_substring = data[index:index + num_timesteps]
		output_substring = data[index+1:index+num_timesteps+1]

		for j in range(num_timesteps):
			input_matrix[i,j,:] = char_to_vector(input_substring[j])
			output_matrix[i,j,:] = char_to_vector(output_substring[j])

	return input_matrix, output_matrix


# Converted this process from a regular method to a class so that object properties
# can be taken advantage of.
class RNN(object):

	def __init__(self, session, scope_name):
		#####################################################################
		#
		# NOTES.
		#  - the softmax output operation reshapes the output from
		#    [batch_size, num_timesteps, num_chars] to 
		#    [batch_size*num_timesteps, num_chars], which means that the
		#    numpy output will have to be reshaped to its original dimensions
		#    to get meaningful output.
		#  - ignore problems, it's working! (thanks to Heng's epic strawberries)
		#
		#####################################################################

		self.session = session

		with tf.variable_scope(scope_name):
			self.weights = tf.Variable(tf.random_normal((num_hidden_units, num_chars), stddev=0.01))
			self.biases = tf.Variable(tf.random_normal((num_chars,), stddev=0.01))

			# Declaring this as a placeholder means that it must be fed with data 
			# at execution time.
			# This means that its value has to be specified using the feed_dict = {}
			# argument inside of Session.run(), Tensor.eval(), or Operation.run()
			# The value that you feed it is going to be a numpy array.
			self.feed_x = tf.placeholder(dtype=tf.float32, shape=(None, None, num_chars))
			self.feed_y = tf.placeholder(dtype=tf.float32, shape=(None, None, num_chars))
			self.feed_lr = tf.placeholder(dtype=tf.float32, shape=(None))
			
			# LSTM's have two sets of states: the cell state, and the hidden state ('c'
			# and 'r'). This list contains lists of c's and r's for each layer of LSTM
			# cells.
			#self.states = tf.placeholder(dtype=tf.float32, shape=(num_lstm_layers, 2, None, num_hidden_units))

			# This converts the placeholders for the states 'r' and 'c' into a tuple 
			# so that they can be passed to the dynamic_rnn. Note that in the case of
			# multiple LSTM layers you'd have N 'r' and 'c' states to pass; one 'r' and
			# 'c' for each layer of the LSTM network.
			self.states = []
			for i in range(num_lstm_layers):
				temp_placeholder_1 = tf.placeholder(dtype=tf.float32, shape=(None, num_hidden_units))
				temp_placeholder_2 = tf.placeholder(dtype=tf.float32, shape=(None, num_hidden_units))
				self.states.append([temp_placeholder_1, temp_placeholder_2])

			self.rnn_tuple_states = []
			for i in range(num_lstm_layers):
				self.rnn_tuple_states.append(tf.contrib.rnn.LSTMStateTuple(self.states[i][0], self.states[i][1]))
			self.rnn_tuple_states = tuple(self.rnn_tuple_states)
			#print("type of states:", type(self.states), type(self.states[0]))

			# Here we assume that every LSTM cell in the network has the same number
			# of hidden nodes.
			self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden_units, forget_bias=1.0)

			# MultiRNNCell - allows multiple recurrent cells to be stacked on top of 
			#   each other. Note that we are explicitly duplicating 
			self.multi_lstm = tf.contrib.rnn.MultiRNNCell([self.lstm_cell]*num_lstm_layers)
			
			self.lstm_zero_states = self.multi_lstm.zero_state(batch_size, dtype=tf.float32)

			# 'outputs' should have dimensions of [batch_size, num_unrolls, num_hidden_units]
			# Since the RNN was defined as being dynamic, the amount of layer unrolling
			# can change from batch to batch. (This script uses a constant batch size).
			# 'last_lstm_state' has dimensions of [batch_size, num_hidden_units]. 
			self.outputs, self.last_lstm_state = tf.nn.dynamic_rnn(cell=self.multi_lstm, inputs=self.feed_x, initial_state=self.rnn_tuple_states, dtype=tf.float32)
			
			# The local field of the softmax output (argument of the softmax).
			# This should return a matrix of dimension [batch_size, num_chars].
			# There's a bias for each output node, and the outputs of the dynamic_rnn are for
			# each timestep, so the output *should* be a matrix of outputs at each timestep.
			local_field = tf.matmul(tf.reshape(self.outputs, [-1, num_hidden_units]), self.weights) + self.biases
			
			# What we're ultimately reducing.
			# This is an operation on the tensorflow graph that contains the 'local_field' object
			# returned from the rnn() method.
			feed_y_long = tf.reshape(self.feed_y, [-1, num_chars])
			self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=local_field, labels=feed_y_long))

			#self.sequence_loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(logits=[local_field], targets=[tf.cast(tf.reshape(feed_y_long, [-1]), tf.int32)], weights=[tf.ones([batch_size * num_timesteps], dtype=tf.float32)])

			outputs_shape = tf.shape(self.outputs)
			# Convert the local fields generated by the output of the LSTM into a softmax output.
			self.softmax_output = tf.reshape(tf.nn.softmax(local_field), (outputs_shape[0], outputs_shape[1], num_chars))

			# Declare the optimizer and the op that you want to minimize.
			self.train_operation = tf.train.RMSPropOptimizer(learning_rate=self.feed_lr).minimize(self.cost)

	def train(self, batch_x, batch_y, lr=0.001):
		
		zero_states = self.session.run(self.lstm_zero_states)
		#print("shape of zero states:", type(zero_states), type(zero_states[0]), zero_states[0][0].shape)
		#zero_states = np.zeros((num_lstm_layers, 2, batch_size, num_hidden_units))
		#zero_state = np.zeros(1, num_hidden_units)
		feed_dict = {}
		feed_dict[self.feed_x] = batch_x
		feed_dict[self.feed_y] = batch_y
		feed_dict[self.feed_lr] = lr
		#feed_dict[self.states] = zero_states
		#print("shape of x, y feeds", batch_x.shape, batch_y.shape)
		
		for i in range(num_lstm_layers):
			feed_dict[self.states[i][0]] = zero_states[i][0]
			feed_dict[self.states[i][1]] = zero_states[i][1]
			#print("shape of zero_states:", zero_states[i][0].shape, zero_states[i][1].shape)
		cost, _ = self.session.run([self.cost, self.train_operation], feed_dict=feed_dict)
		
		return cost

	def test(self, valid_x, valid_y):
		zero_states = self.session.run(self.lstm_zero_states)

		feed_dict = {}
		feed_dict[self.feed_x] = valid_x
		feed_dict[self.feed_y] = valid_y

		for i in range(num_lstm_layers):
			feed_dict[self.states[i][0]] = zero_states[i][0]
			feed_dict[self.states[i][1]] = zero_states[i][1]

		cost = self.session.run(self.cost, feed_dict=feed_dict)
		perplexity = np.exp(cost)
		#print("perplexity", np.exp(cost))
		#input()
		return perplexity

	def run(self, x, num_steps=25, delimiter=None):
		'''
		This method will create 'num_steps' unrollings of the network after passing
		the input tensor, 'x', to the network.
		'''
		test_output = ""
		#zero_state = self.session.run(self.multi_lstm.zero_state(x.shape[0], dtype=tf.float32))
		zero_states = np.zeros((num_lstm_layers, 2, 1, num_hidden_units))
		lstm_next_state = zero_states
		#feed_dict={self.feed_x:x, self.state_r:zero_state_r, self.state_c:zero_state_c}
		feed_dict={}

		for i in range(x.shape[0]):
			feed_dict[self.feed_x] = [[x[i]]]
			#feed_dict[self.states] = lstm_next_state
			for j in range(num_lstm_layers):
				feed_dict[self.states[j][0]] = lstm_next_state[j][0]
				feed_dict[self.states[j][1]] = lstm_next_state[j][1]
				#print(str(lstm_next_state[j][0]))
				#print(str(lstm_next_state[j][1]))
			softmax_out, lstm_state_out = self.session.run([self.softmax_output, self.last_lstm_state], feed_dict=feed_dict)
			lstm_next_state = lstm_state_out
			#print("lstm state type, size:", type(lstm_state_out), lstm_state_out[0][0].shape)
			char_out = vector_to_char(softmax_out[0][0])
			test_output += char_out
			
		for i in range(num_steps):
			lstm_in = char_to_vector(char_out)
			lstm_next_state = lstm_state_out

			feed_dict[self.feed_x] = [[lstm_in]]
			#feed_dict[self.states] = lstm_next_state
			for j in range(num_lstm_layers):
				feed_dict[self.states[j][0]] = lstm_next_state[j][0]
				feed_dict[self.states[j][1]] = lstm_next_state[j][1]
			softmax_out, lstm_state_out = self.session.run([self.softmax_output, self.last_lstm_state], feed_dict=feed_dict)	
			#lstm_next_state = lstm_state_out
			char_out = vector_to_char(softmax_out[0][0])
			test_output += char_out

		return test_output

def main(argv):
	load_checkpoint = False
	num_plateaus = 0
	perplexity = []
	halving_threshold = 0.003
	learning_rate = 0.1
	max_plateaus = 5

	date = datetime.now()
	date = str(date).split()[0]

	if len(argv) > 0:
		load_checkpoint = True
		checkpoint_file = argv[0]
	# Start tf session. This is passed to the RNN class.
	sess = tf.Session()

	cud = os.getcwd()
	weight_saver = thread_ops.weightThread()

	network = RNN(sess, 'stuff')

	# Initialize all of the variables inside of the 'sess' tensorflow session.
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	if load_checkpoint and os.path.isfile(os.path.join(cud, checkpoint_file)):
	#if False:
		saver.restore(sess, os.path.join(cud, "lstmsmall.ckpt"))
		print(network.run(string_to_tensor('random stuff')))

	else:
		step = 1

		# Retrieve the list of trainable variables. The goal is to save these
		# weights as a grayscale PNG to show how the values evolve over time.
		trainable_vars = tf.trainable_variables()
		
		while step < max_steps and num_plateaus < max_plateaus:
			# The training batch has to have specific dimensions:
			#   (batch_size, num_unrolls, vocab_length)
			# num_unrolls - number of characters in the training sequence
			# vocab_length - number of unique items in the training vocabulary
			# Since we're building a language model, the training_batch is also the
			# desired outputs, except the desired outputs are shifted by one word.
			train_x, train_y = get_random_batch(num_timesteps, batch_size, train_data)

			valid_x, valid_y = get_random_batch(num_timesteps, batch_size, valid_data)
			#print(step)
			# Note that the elements of the feed dictionary are the placeholders
			# defined earlier in the program.
			cost = network.train(train_x, train_y, lr=learning_rate)
			
			if step % int(sample_step_percentage*float(max_steps)) == 0:
				print("-----------------------------------------------------")
				ppl = network.test(valid_x, valid_y)
				perplexity.append(ppl)
				print(float(step)*100.0/float(max_steps), "percent complete")
				print("current step", step, "out of", max_steps)
				print("\tcost:", cost)
				print("\tperplexity:", ppl)
				#print("\tnumber of weight images saved:", weight_saver.iteration)
				training_output = network.run(string_to_tensor('the '), num_steps=500)
				if len(perplexity) > 1:
					dp = perplexity[-2] - perplexity[-1]
					dp /= perplexity[-2]
					print("\tdelta_perplexity:", dp)
					if dp < halving_threshold:
						learning_rate /= 2.
						print("\t\tlearning rate halved", learning_rate)
						num_plateaus += 1
					dp = 0
				print("training output:\n", training_output)
				#print("sequence loss:\n", sequence_loss)
				#print("local field:\n", sess.run(network.local_field))
				

			if step % int(sample_weight_percentage*float(max_steps)) == 0:
				with open(date + 'train_errors.txt', 'a') as f:
					f.write(str(cost) + '\n')
				#weights = sess.run(trainable_vars)
				#weight_saver.run(weights)

			step += 1

		saver.save(sess, os.path.join(cud, "lstmsmall.ckpt"))
	print(network.run(string_to_tensor('random stuff')))

if __name__ == "__main__":
	main(sys.argv[1:])
