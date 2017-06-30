import numpy as np
import cv2
import ConvertToMNIST
from PIL import Image, ImageFilter
import random

images = []
labels = []
for i in range(0, 10):
	for j in range(0, 500):		
		temp = cv2.imread("ComputerMNIST\\"  + str(i) + "\\" + str(j) + ".jpg")
		temp = ConvertToMNIST.clusterImage(temp)
		temp = ConvertToMNIST.imageprepare(Image.fromarray(temp))
		images += [temp]
		temp = [0,0,0,0,0,0,0,0,0,0]
		temp[i] = 1
		labels+=[temp]
 	
shuffle_indexes = []
print(len(images))
for i in range(0, len(images)):
	shuffle_indexes.append(i)
random.shuffle(shuffle_indexes)
images_shuf = []
labels_shuf = []
for i in shuffle_indexes:
	images_shuf +=[images[i]]
	labels_shuf +=[labels[i]]
images_shuf = np.array(images_shuf)
labels_shuf = np.array(labels_shuf)
np.save("ComputerMNISTImages.npy", images_shuf)
np.save("ComputerMNISTLabels.npy", labels_shuf)