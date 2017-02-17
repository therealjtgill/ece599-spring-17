import tensorflow as tf
from tensorflow.contrib import rnn

# Need to gather the training samples so the max number of training steps can
# be defined.

num_chars = 28
num_timesteps = 20
num_hidden_units = 200
learning_rate = 0.001
batch_size = 32
max_epochs = 30
max_steps = max_epochs * batch_size / num_training_samples
sample_step = 100
# The maximum number of steps you go through is 
#   max_epochs * batch_size / num_training_samples

weights = {'out':tf.Variable(tf.random_normal([num_hidden_units, num_chars]))}
biases = {'out': tf.Variable(tf.random_normal([num_chars]))}

# Declaring this as a placeholder means that it must be fed with data 
# at execution time.
# This means that its value has to be specified using the feed_dict = {}
# argument inside of Session.run(), Tensor.eval(), or Operation.run()
# The value that you feed it is going to be a numpy array.
feed_x = tf.placeholder(dtype=tf.float32, shape=(None, num_chars))
feed_y = tf.placeholder(dtype=tf.float32, shape=(None, num_chars))

def rnn(input_vector, w, b):
	# The cell state and the hidden state are vectors with the same number
	# of components. This is usually coupled with an op that constructs the
	# actual network - rnn.static_rnn, dynamic_rnn, etc.
	lstm_cell = rnn.BasicLSTMCell(num_units=num_hidden_units, forget_bias=1.0, state_is_tuple=True)

	# 'output' should have dimensions of [batch_size, num_unrolls, num_hidden_units]
	# Since the RNN was defined as being dynamic, the amount of layer unrolling
	# can change from batch to batch.
	output, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=input_vector,dtype=tf.float32)

	local_field = tf.matmul(tf.reshape(outputs, [-1, num_hidden_units]), w) + b

	return local_field

def get_next_batch(batch_size):
	

local_field = rnn(feed_x, weights['out'], biases['out'])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=local_field, labels=feed_y))

softmax_output = tf.nn.softmax(local_field)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	# Initialize all of the variables we've declared to their default values.
	# In this context, you initialize the weights and biases to random
	# normal values.
	sess.run(init)

	# Recall that one epoch is a forward pass and backward pass for every 
	# training sample.
	step = 1
	
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
		# desired outputs.
		train_batch = get_next_batch(batch_size)

		# Note that the elements of the feed dictionary are the placeholders
		# defined earlier in the program.
		sess.run(optimizer, feed_dict={feed_x: train_batch, feed_y: train_batch})

		if steps % sample_step == 0:

			print(sess.run(softmax_output, feed_dict={feed_x: train_batch}))