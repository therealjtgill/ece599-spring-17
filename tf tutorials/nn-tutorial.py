import tensorflow as tf
from tensorflow.contrib import rnn

num_chars = 28
num_timesteps = 20
num_hidden_units = 200
learning_rate = 0.001
batch_size = 32

weights = {'out':tf.Variable(tf.random_normal([num_chars, num_hidden_units]))}
biases = {'out': tf.Variable(tf.random_normal([num_chars]))}


# Declaring this as a placeholder means that it must be fed with data 
# at execution time.
# This means that its value has to be specified using the feed_dict = {}
# argument inside of Session.run(), Tensor.eval(), or Operation.run()
# The value that you feed it is most likely going to be a numpy array.
feed_x = tf.placeholder(dtype=tf.float32, shape=(None, num_chars))
feed_y = tf.placeholder(dtype=tf.float32, shape=(None, num_chars))

def rnn(input_vector, weights, biases):
	# The cell state and the hidden state are vectors with the same number
	# of components. This is usually coupled with an op that constructs the
	# actual network - rnn.static_rnn, dynamic_rnn, etc.
	lstm_cell = rnn.BasicLSTMCell(num_units=num_hidden_units, forget_bias=1.0, state_is_tuple=True)
	
	# Outputs should have dimensions of [num_unrolls, num_hidden_units]
	# Since the RNN was defined as being dynamic, the amount of layer unrolling
	# can change from batch to batch.
	outputs, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=input_vector,dtype=tf.float32)


	return outputs