import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
def imageprepare(im):
  #im = Image.open(argv).convert('L')
  width = float(im.size[0])
  height = float(im.size[1])
  newImage = Image.new('L', (28, 28), (255)) 
  
  if width > height: 
    nheight = int(round((20.0/width*height),0)) 
    if (nheigth == 0): 
      nheigth = 1  
    img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    wtop = int(round(((28 - nheight)/2),0)) 
    newImage.paste(img, (4, wtop))
  else:
    nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
    if (nwidth == 0): #rare case but minimum is 1 pixel
      nwidth = 1
     # resize and sharpen
    img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
    newImage.paste(img, (wleft, 4)) #paste resized image on white canvas
  
  tv = list(newImage.getdata()) #get pixel values
  
  #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
  tva = [ (255-x)*1.0/255.0 for x in tv] 
  return tva

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

def clusterImage(img):
    convert = img.reshape((img.shape[0]*img.shape[1],3))
    convert = np.float32(convert)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(convert,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    if(difference(center[0], img[0][0]) > difference(center[1], img[0][0])):
    	center[1] = [255.0, 255.0, 255.0]
    	center[0] = [0, 0, 0]
    else:
    	center[0] = [255.0, 255.0, 255.0]
    	center[1] = [0, 0, 0]
    ans = center[label.flatten()]
    res2 = ans.reshape((img.shape))
    return res2

def difference(cluster, pixel):
	c = np.int16(cluster)
	p = np.int16(pixel)

	return abs(c[0] - p[0] ) + abs(c[1]-p[1]) + abs(c[2] - p[2])

