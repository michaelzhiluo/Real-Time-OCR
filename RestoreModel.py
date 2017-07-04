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
saver = tf.train.import_meta_graph('MyModel\\CNNModel.meta')
saver.restore(sess,tf.train.latest_checkpoint('MyModel'))

graph = tf.get_default_graph()
training_data = graph.get_tensor_by_name("training_data:0")
training_labels = graph.get_tensor_by_name("training_labels:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")

predict = graph.get_tensor_by_name("predicted_number:0")

counter =11000
roi = []
# 0 -> x1, 1 -> x2, 2 -> y1, 3-> y2
roi += [[320, 345, 115, 175]]
roi += [[345, 373, 115, 175]]
roi += [[372, 399, 115, 175]]
roi += [[398, 427, 115, 175]]
roi += [[438, 465, 115, 175]]
roi += [[465, 491, 115, 175]]
roi += [[268, 292, 305, 365]]
roi += [[292, 318, 305, 365]]
roi += [[319, 344, 305, 365]]
roi += [[345, 368, 305, 365]]
roi += [[370, 395, 305, 365]]
roi += [[396, 423, 305, 365]]
roi += [[435, 462, 305, 365]]
roi += [[463, 487, 305, 365]]

while(counter <=20000):
	frame =  cv2.imread("C:\\Users\\Michael Luo\\Documents\\ComputerDigitsTrainingImages\\" + str(counter) + ".jpg");
	prediction = [] 
	for i in range(0, len(roi)):
		temp = frame[roi[i][2]: roi[i][3], roi[i][0]: roi[i][1]]
		temp = ConvertToMNIST.clusterImage(temp)
		temp = ConvertToMNIST.imageprepare(Image.fromarray(temp))
		pred = sess.run(predict, feed_dict ={training_data: [temp] ,training_labels: [[1,0,0,0,0,0,0,0,0,0]], keep_prob: 1})
		prediction += [pred[0]]

	for i in range(0, len(roi)):
		cv2.rectangle(frame, (roi[i][0], roi[i][2]), (roi[i][1], roi[i][3]), (255, 255, 255))
		cv2.putText(frame, str(prediction[i]), (roi[i][0] - 10, roi[i][2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	counter+=1


cv2.destroyAllWindows()


"""
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
"""
"""
temp = cv2.imread("C:\\Users\\Michael Luo\\Documents\\TrainingData\\108761.jpg")
temp = ConvertToMNIST.clusterImage(temp)
temp = ConvertToMNIST.imageprepare(Image.fromarray(temp))
prediction = sess.run(predict, feed_dict ={training_data: [temp] ,training_labels: [[0,0,0,0,0,0,0,0,0,1]], keep_prob: 1})
print(prediction)
"""