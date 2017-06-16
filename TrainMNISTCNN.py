from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Input Data and Labels
training_data = tf.placeholder(tf.float32, [None, 784], name = "training_data")
training_labels = tf.placeholder(tf.float32, [None, 10], name = "training_labels")
keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

# Defining Variables
layer1 = tf.Variable(tf.random_normal([5,5,1,32]))
bias1 = tf.Variable(tf.random_normal([32]))

# 1st Convoluted Layer
x = tf.reshape(training_data, shape = [-1, 28, 28, 1])
conv1 = tf.nn.conv2d(x, layer1, strides = [1,1,1,1], padding = 'SAME')
b1 = tf.nn.bias_add(conv1, bias1)
actv1 = tf.nn.relu(b1)

#1st Pooling Layer
pool1 = tf.nn.max_pool(actv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

# Defining Variables
layer2 = tf.Variable(tf.random_normal([5,5,32,64]))
bias2 = tf.Variable(tf.random_normal([64]))

# 2nd Convoluted Layer
conv2 = tf.nn.conv2d(pool1, layer2, strides =[1,1,1,1], padding = 'SAME')
b2 = tf.nn.bias_add(conv2, bias2)
actv2 = tf.nn.relu(b2)

#2nd Pooling Layer
pool2 = tf.nn.max_pool(actv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

#Defining Variables
layer3 = tf.Variable(tf.random_normal([7*7*64, 1024]))
bias3 = tf.Variable(tf.random_normal([1024]))

#Fully-Connected Layer
reshapef = tf.reshape(pool2, [-1, 7*7*64])
add1 = tf.add(tf.matmul(reshapef, layer3), bias3)
actv3 = tf.nn.relu(add1)
final = tf.nn.dropout(actv3, keep_prob)

#Converting to class scores
layer4 = tf.Variable(tf.random_normal([1024, 10]))
bias4 = tf.Variable(tf.random_normal([10]))

output =  tf.add(tf.matmul(final, layer4), bias4)

# Calculating Loss Function
loss_vector = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=training_labels)
cost = tf.reduce_mean(loss_vector)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
predicted_number = tf.argmax(output, 1, name = "predicted_number");
correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(training_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#---------------------------------------------------------------------------------------------------------------------------#

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    
    # Keep training until reach max iterations
    while step * 128 < 200000:
        batch_x, batch_y = mnist.train.next_batch(128)
        sess.run(optimizer, feed_dict={training_data: batch_x, training_labels: batch_y, keep_prob: 0.75})
        
        if(step%100==0):
            loss, acc = sess.run([cost, accuracy], feed_dict={training_data: batch_x,
                                                           training_labels: batch_y, keep_prob: 1
                                                              })
            print("Generation " + str(step*128) + " with normal batch, Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
        
    
    ltr = saver.save(sess, "\\MyModel\\CNNModel")
    data = mnist.test.images[:128];
    labels = mnist.test.labels[:128];
    
    pred, ac = sess.run([predicted_number, accuracy], feed_dict={training_data: data
                                      ,training_labels: labels,
                                      keep_prob: 1.})
    print(pred)
    print(np.argmax(labels, 1))
    print(ac)