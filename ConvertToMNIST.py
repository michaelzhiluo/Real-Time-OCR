import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def convertImageToMNIST(img):
	resize = cv2.resize(img, dsize =(28, 28))
	gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
	average = cv2.mean(gray)
	if(gray[0][0] < average[0]):
		gray = 255 - gray
	_,thres = cv2.threshold(gray, 255-average[0], 0, 2)
	num = np.asarray(thres, np.float32)
	num = 255 -num
	num = (num -np.min(num))/(np.max(num) - np.min(num))
	return num

def clusterImage(img, numclusters):
    convert = img.reshape((img.shape[0]*img.shape[1],3))
    convert = np.float32(convert)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(convert,numclusters,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    ans = center[label.flatten()]
    res2 = ans.reshape((img.shape))
    cv2.imshow("hi", res2)
    cv2.waitKey(0);
    return res2

