import numpy as np

class DataHandler(object):
	def __init__(self, train_file, test_file, validation_file):

		with open(train_file) as f:
			self.train_data = f.read()
		self.train_data = self.train_data.lower()

		with open(test_file) as f:
			self.test_data = f.read()
		self.test_data = self.test_data.lower()

		with open(validation_file) as f:
			self.validation_data = f.read()
		self.validation_data = self.validation_data.lower()

		all_data = self.train_data + self.validation_data + self.test_data
		self.char_vocab = list(set(all_data))
		self.vocab_size = len(self.char_vocab)

		self.dataset = {
			'train':self.train_data,
			'validation':self.validation_data,
			'test':self.test_data
		}

	def char_to_vector(self, char):
		char = char.lower()
		vector = np.zeros(self.vocab_size)
		vector[self.char_vocab.index(char)] = 1.0
		return vector

	def vector_to_char(self, vector):
		index = np.random.choice(range(self.vocab_size), p=vector)
		return self.char_vocab[index]

	def string_to_tensor(self, string_):
		'''
		Converts a string into a tensor (numpy array). This is intended for running the
		network on a sequence of characters.
		The output will have the shape
		[string_length, vocabulary_size]
		'''
		tensor = np.zeros((len(string_), self.vocab_size))
		for j in range(len(string_)):
			tensor[j,:] = self.char_to_vector(string_[j])
		return tensor

	def get_random_batch(self, num_timesteps, batch_size, dataset):
		data = self.dataset[dataset]

		input_matrix = np.zeros((batch_size, num_timesteps, self.vocab_size))
		output_matrix = np.zeros((batch_size, num_timesteps, self.vocab_size))

		for i in range(batch_size):
			index = int((len(data)-num_timesteps-1)*np.random.rand())
			input_substring = data[index:index+num_timesteps]
			output_substring = data[index+1:index+num_timesteps+1]

			for j in range(num_timesteps):
				input_matrix[i,j,:] = self.char_to_vector(input_substring[j])
				output_matrix[i,j,:] = self.char_to_vector(output_substring[j])

		return input_matrix, output_matrix