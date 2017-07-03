import numpy as np


class ComputerMNIST:

	def ComputerMNIST(self):
		self.images = np.load("ComputerMNISTImages.npy")
		self.labels = np.load("ComputerMNISTLabels.npy")
		self.num_examples = len(images)

	def next_batch(batch_size):


	@property
	def images(self):
		return self.images

	@property
	def labels(self):
		return self.labels

