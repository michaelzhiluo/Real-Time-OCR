import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import ConvertToMNIST
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
from PIL import Image, ImageFilter

def imageprepare(im):
  #im = Image.open(argv).convert('L')
  print(im.size)
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

sess=tf.Session()    
saver = tf.train.import_meta_graph('C:\\MyModel\\CNNModel.meta')
saver.restore(sess,tf.train.latest_checkpoint('C:\\MyModel'))

graph = tf.get_default_graph()
training_data = graph.get_tensor_by_name("training_data:0")
training_labels = graph.get_tensor_by_name("training_labels:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")

predict = graph.get_tensor_by_name("predicted_number:0")

temp = cv2.imread("997.jpg")
temp = ConvertToMNIST.clusterImage(temp, 3)
temp = ConvertToMNIST.convertImageToMNIST(temp)
#temp = imageprepare(temp)
#temp = mnist.test.images[0].reshape((28, 28))
#matrix = ConvertToMNIST.clusterImage(img, 2)
#print(matrix)
#temp = ConvertToMNIST.convertImageToMNIST(matrix)
#temp = mnist.test.images[0].reshape((28, 28))
#for i in range(0, 28):
	#for j in range(0, 28):
		#print("%5.3f" % temp[i][j], end = '')
	#print("\n")
#prediction = sess.run(predict, feed_dict ={training_data: [temp.flatten()] ,training_labels: [[0,0,0,0,0,0,0,0,0,1]], keep_prob: 1})
#print(prediction)

plt.imshow(np.array(temp).reshape((28, 28)), cmap = 'gray')
plt.show()


'''
video = cv2.VideoCapture(0)
while(True):
    ret, frame = video.read()   
    
    cv2.imshow('test', frame)
    cv2.waitKey(5)

'''
