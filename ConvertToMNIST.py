import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def clusterImage(image, numclusters):
	convert = image.reshape((image.shape[0]*image.shape[1],3))
	convert = np.float32(convert)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret,label,center=cv2.kmeans(convert,numclusters,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	center = np.uint8(center)
	ans = center[label.flatten()]
	res2 = ans.reshape((image.shape))
	return res2
