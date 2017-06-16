import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
sess=tf.Session()    
saver = tf.train.import_meta_graph('C:\\MyModel\\CNNModel.meta')
saver.restore(sess,tf.train.latest_checkpoint('C:\\MyModel'))

graph = tf.get_default_graph()
training_data = graph.get_tensor_by_name("training_data:0")
training_labels = graph.get_tensor_by_name("training_labels:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")

predict = graph.get_tensor_by_name("predicted_number:0")

originalimg = cv2.imread("Three.jpg")
resize = cv2.resize(originalimg, dsize =(28, 28))
gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
print(gray)
average = cv2.mean(gray)
if(gray[0][0] < average[0]):
    gray = 255 - gray
_,thres = cv2.threshold(gray, 255-average[0], 0, 2)
print(average[0])
nump = np.asarray(thres, np.float32)
nump = 255 - nump
nump = (nump -np.min(nump))/(np.max(nump) - np.min(nump))
plt.imshow(nump, cmap = 'gray')
plt.show()
plt.imshow(mnist.test.images[0].reshape(28, 28), cmap = 'gray')
plt.show()

prediction = sess.run(predict, feed_dict ={training_data: [nump.flatten()] ,training_labels: [[0,0,0,0,0,0,0,0,0,1]], keep_prob: 1})
print(prediction)

'''
video = cv2.VideoCapture(0)
while(True):
    ret, frame = video.read()   
    
    cv2.imshow('test', frame)
    cv2.waitKey(5)

'''
