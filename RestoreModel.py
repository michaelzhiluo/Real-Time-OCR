import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import ConvertToMNIST
import os
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
from PIL import Image, ImageFilter

"""
	RestoreModel
	Runs the webcame images and restored neural network model to identify webcam images.
"""
refPt = []
uppercorner =[]
cropping = False
 
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping, uppercorner
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		uppercorner = (x, y)
		cropping = True
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt += [[uppercorner, (x,y)]]
		cropping = False
	if cropping:
 		cv2.rectangle(img, uppercorner, (x,y), (0, 255, 0), 1)
 		cv2.imshow("input", img)
 

sess=tf.Session()    
#s1 = input("Pathname Folder where CNN Model is stored (Meta file must be named \"CNNModel.meta\"):")
#saver = tf.train.import_meta_graph(s1 + "\\CNNModel.meta")
#saver.restore(sess,tf.train.latest_checkpoint(s1))
saver = tf.train.import_meta_graph('MyModel\\CNNModel.meta')
saver.restore(sess,tf.train.latest_checkpoint('MyModel'))

graph = tf.get_default_graph()
training_data = graph.get_tensor_by_name("training_data:0")
training_labels = graph.get_tensor_by_name("training_labels:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")

predict = graph.get_tensor_by_name("predicted_number:0")


cap = cv2.VideoCapture(0)
cv2.namedWindow("input")
cv2.setMouseCallback("input", click_and_crop)
print("Select Region of Interests from the Webcam:")
print("(Press Q to Finish ROI Selection)")
while True:
	ret, img = cap.read()
	cv2.imshow("input", img)
	for i in refPt:
		cv2.rectangle(img, i[0], i[1], (0, 255, 0), 1)
		cv2.imshow("input", img)
	key = cv2.waitKey(1)
	if key == ord("q"):
		break

cv2.destroyAllWindows();

roi = []
for i in refPt:
	lel = [i[0][0], i[1][0], i[0][1], i[1][1]]
	roi += [lel]

while True:
	ret, img = cap.read()
	#frame =  cv2.imread("C:\\Users\\Michael Luo\\Documents\\WebcamImages\\" + str(counter) + ".jpg")
	prediction = [] 
	for i in range(0, len(roi)):
		temp = img[roi[i][2]: roi[i][3], roi[i][0]: roi[i][1]]
		temp = ConvertToMNIST.clusterImage(temp)
		temp = ConvertToMNIST.imageprepare(Image.fromarray(temp))
		plt.plot(temp)
		pred = sess.run(predict, feed_dict ={training_data: [temp] ,training_labels: [[1,0,0,0,0,0,0,0,0,0]], keep_prob: 1})
		prediction += [pred[0]]

	for i in range(0, len(roi)):
		cv2.rectangle(img, (roi[i][0], roi[i][2]), (roi[i][1], roi[i][3]), (255, 255, 255))
		cv2.putText(img, str(prediction[i]), (roi[i][0] - 10, roi[i][2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
	cv2.imshow('input',img)
	if cv2.waitKey(100) & 0xFF == ord('q'):
		break
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