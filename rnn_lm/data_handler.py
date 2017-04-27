import numpy as np
import os

# Removed this from the data handler class. Using the entire ASCII table makes
# this object-independent.
def char_to_vector(char):
	char = char.lower()
	vector = np.zeros(DataHandler.vocab_size)
	if char in DataHandler.vocab:
		vector[DataHandler.vocab.index(char)] = 1.0
	else:
		vector[DataHandler.vocab.index('@')] = 1.0
	return vector

# Removed this from the data handler class. Using the entire ASCII table makes
# this object-independent.
def vector_to_char(vector):
	index = np.random.choice(range(DataHandler.vocab_size), p=vector)
	#index = np.where(vector == np.amax(vector))[0][0]
	return DataHandler.vocab[index]

def string_to_tensor(string_):
	'''
	Converts a string into a tensor (numpy array). This is intended for running the
	network on a sequence of characters.
	The output will have the shape [string_length, vocab_size]
	'''
	tensor = np.zeros((len(string_), DataHandler.vocab_size))
	for j in range(len(string_)):
		tensor[j,:] = char_to_vector(string_[j])
	return tensor

def prob_to_char(prob):
	return vector_to_char(prob)

def probs_to_string(probs):
	message = ''
	for prob in probs:
		message += prob_to_char(prob)

	return message

def soft_prob_to_one_hot(prob):
	index = np.random.choice(range(DataHandler.vocab_size), p=prob)
	one_hot = np.zeros_like(prob)
	one_hot[index] = 1.

	return one_hot

class DataHandler(object):

	vocab = [chr(x) for x in range(128)]
	vocab_size = len(vocab)

	def __init__(self, train_file, test_file, validation_file, data_dir=''):

		with open(os.path.join(data_dir, train_file)) as f:
			self.train_data = f.read()
		self.train_data = self.train_data.lower()

		with open(os.path.join(data_dir, test_file)) as f:
			self.test_data = f.read()
		self.test_data = self.test_data.lower()

		with open(os.path.join(data_dir, validation_file)) as f:
			self.validation_data = f.read()
		self.validation_data = self.validation_data.lower()

		all_data = self.train_data + self.validation_data + self.test_data + "@"
		#self.vocab = list(set(all_data))

		self.dataset = {
			'train':self.train_data,
			'validation':self.validation_data,
			'test':self.test_data
		}

	def get_random_batch(self, num_timesteps, batch_size, dataset):
		data = self.dataset[dataset]

		input_matrix = np.zeros((batch_size, num_timesteps, self.vocab_size))
		output_matrix = np.zeros((batch_size, num_timesteps, self.vocab_size))

		for i in range(batch_size):
			index = int((len(data)-num_timesteps-1)*np.random.rand())
			input_substring = data[index:index+num_timesteps]
			output_substring = data[index+1:index+num_timesteps+1]

			for j in range(num_timesteps):
				input_matrix[i,j,:] = char_to_vector(input_substring[j])
				output_matrix[i,j,:] = char_to_vector(output_substring[j])

		return input_matrix, output_matrix

	@property
	def name(self):
		return self._name

	@name.setter
	def name(self, name):
		self._name = name