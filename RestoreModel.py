import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import ConvertToMNIST
import os
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
from PIL import Image, ImageFilter
sess=tf.Session()    
saver = tf.train.import_meta_graph('C:\\MyModel\\CNNModel.meta')
saver.restore(sess,tf.train.latest_checkpoint('C:\\MyModel'))

graph = tf.get_default_graph()
training_data = graph.get_tensor_by_name("training_data:0")
training_labels = graph.get_tensor_by_name("training_labels:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")

predict = graph.get_tensor_by_name("predicted_number:0")
counter  =0
for i in range(50, 2000):
	if(counter ==50):
		break
	temp = cv2.imread("C:\\Users\\Michael Luo\\Documents\\TrainingData\\" + str(i) + ".jpg")
	if(temp == None): 
		continue
	print(str(i) + ".jpg")
	temp = ConvertToMNIST.clusterImage(temp, 2)
	temp = ConvertToMNIST.imageprepare(Image.fromarray(temp))
	
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
	prediction = sess.run(predict, feed_dict ={training_data: [temp] ,training_labels: [[0,0,0,0,0,0,0,0,0,1]], keep_prob: 1})
	print(prediction)
	#print(temp)
	counter+=1
	#plt.imshow(np.array(temp).reshape((28, 28)), cmap = 'gray')
	#plt.show()