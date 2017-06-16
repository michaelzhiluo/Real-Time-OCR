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

t_data = mnist.test.images[:128]
t_label = mnist.test.labels[:128]

prediction = sess.run(predict, feed_dict ={training_data: t_data ,training_labels: t_label, keep_prob: 1})
print(prediction)
temp = np.argmax(t_label, 1) 
print(temp)
print(np.equal(prediction, temp))