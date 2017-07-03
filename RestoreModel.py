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
saver = tf.train.import_meta_graph('\\MyModel\\CNNModel.meta')
saver.restore(sess,tf.train.latest_checkpoint('\\MyModel'))

graph = tf.get_default_graph()
training_data = graph.get_tensor_by_name("training_data:0")
training_labels = graph.get_tensor_by_name("training_labels:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")

predict = graph.get_tensor_by_name("predicted_number:0")



counter =0
gg =0
for j in range(0, 10):
	counter =0
	for i in range(0, 500):
		temp = cv2.imread("C:\\Users\\Michael Luo\\Documents\\ComputerMNIST\\"  + str(j) + "\\" + str(i) + ".jpg")
		temp = ConvertToMNIST.clusterImage(temp)
		temp = ConvertToMNIST.imageprepare(Image.fromarray(temp))
		prediction = sess.run(predict, feed_dict ={training_data: [temp] ,training_labels: [[0,0,0,0,0,0,0,0,0,1]], keep_prob: 1})
		if(prediction == j):
			counter+=1
	print(j)
	print(str(counter) + "/500")
	gg+=counter

print("total accuracy:" + str(gg/5000.0))
