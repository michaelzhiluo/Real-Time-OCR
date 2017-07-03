import numpy as np


class ComputerMNIST(object):

	def __init__(self):
		self._images = np.load("ComputerMNISTImages.npy")
		self._labels = np.load("ComputerMNISTLabels.npy")
		self.num_examples = len(self._images)
		self.num_epochs =0
		self.index_in_epoch =0

	def next_batch(self, batch_size):
		if self.index_in_epoch + batch_size > self.num_examples:
			self.num_epochs+=1

			#Images and Labels before the end of epoch
			images1 = self._images[self.index_in_epoch:self.num_examples]
			labels1 = self._labels[self.index_in_epoch: self.num_examples]

			#Shuffle Data after Epoch is complete
			perm = np.arange(self.num_examples)
			np.random.shuffle(perm)
			self._images = self.images[perm]
			self._labels = self.labels[perm]

        	#Images and Labels in the new epoch
			batch_size -= self.num_examples - self.index_in_epoch
			self.index_in_epoch = batch_size
			images2 = self._images[0: self.index_in_epoch]
			labels2 = self._labels[0: self.index_in_epoch]

			return np.concatenate((images1, images2), axis =0), np.concatenate((labels1, labels2), axis =0)
		
		start = self.index_in_epoch
		self.index_in_epoch += batch_size
		return self._images[start: self.index_in_epoch], self._labels[start: self.index_in_epoch]

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels