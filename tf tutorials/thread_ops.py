import threading
import os
import scipy.misc
import numpy as np
from math import sqrt

class weightThread(threading.Thread):
	def __init__(self):
		self.iteration = 0


	def run(self, weights):
		print "Starting weight saving thread..."
		#threadLock.acquire()
		self.iteration += 1
		threading.Thread(target=save_weights_as_png, args=(weights,self.iteration))
		outfilename = save_weights_as_png(weights, self.iteration)
		#threadLock.release()
		print "Weights successfully written to file", outfilename

def save_weights_as_png(weights, iteration):
	flat_weights = weights[0].flatten()
	for weight in weights[1:]:
		print("flattened shapes", flat_weights.shape, weight.flatten().shape)
		flat_weights = np.concatenate((flat_weights, weight.flatten()), axis=0)
	if not int(sqrt(len(flat_weights))) ** 2 == len(flat_weights):
		padding = abs(int(sqrt(len(flat_weights))+1) ** 2 - len(flat_weights))
		flat_weights = np.concatenate((flat_weights, np.zeros(padding)), axis=0)
	image_dim = int(sqrt(len(flat_weights)))
	print("flat weights shape", flat_weights.shape)
	print(image_dim)
	square_weights = flat_weights.reshape(image_dim, image_dim)
	print("flat weights reshaped shape", square_weights.shape)
	filename = 'weights' + str(iteration) + '.png'
	scipy.misc.imsave(os.path.join(os.getcwd(), filename), square_weights)
	return filename