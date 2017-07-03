from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ComputerMNIST

filter_height_conv_layer_1 = 5
filter_width_conv_layer_1 = 5
layer_1_channels = 32
stride_conv_layer_1 = 1
pool_kernel_size_layer_1 = 2
pool_strides_layer_1 = 2

filter_height_conv_layer_2 = 5
filter_width_conv_layer_2 = 5
layer_2_channels = 64
stride_conv_layer_2 = 1
pool_kernel_size_layer_2 = 2
pool_strides_layer_2 = 2

layer_3_channels = 1024

num_iterations = 200000
batch_size = 128
rate_learning = 0.001
dropout_prob = 0.75
test_batch_size = 512
ComputerMNIST_batch_size = 50
case =2

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Input Data and Labels
training_data = tf.placeholder(tf.float32, [None, 784], name = "training_data")
training_labels = tf.placeholder(tf.float32, [None, 10], name = "training_labels")
keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

# Defining Variables
layer1 = tf.Variable(tf.random_normal([filter_height_conv_layer_1,filter_width_conv_layer_1,1,layer_1_channels]))
bias1 = tf.Variable(tf.random_normal([layer_1_channels]))

# 1st Convoluted Layer
x = tf.reshape(training_data, shape = [-1, 28, 28, 1])
conv1 = tf.nn.conv2d(x, layer1, strides = [1,stride_conv_layer_1,stride_conv_layer_1,1], padding = 'SAME')
b1 = tf.nn.bias_add(conv1, bias1)
actv1 = tf.nn.relu(b1)

#1st Pooling Layer
pool1 = tf.nn.max_pool(actv1, ksize = [1,pool_kernel_size_layer_1,pool_kernel_size_layer_1,1], strides = [1,pool_strides_layer_1,pool_strides_layer_1,1], padding = 'SAME')

# Defining Variables
layer2 = tf.Variable(tf.random_normal([filter_height_conv_layer_2, filter_width_conv_layer_2, layer_1_channels,layer_2_channels]))
bias2 = tf.Variable(tf.random_normal([layer_2_channels]))

# 2nd Convoluted Layer
conv2 = tf.nn.conv2d(pool1, layer2, strides =[1,stride_conv_layer_2,stride_conv_layer_2,1], padding = 'SAME')
b2 = tf.nn.bias_add(conv2, bias2)
actv2 = tf.nn.relu(b2)

#2nd Pooling Layer
pool2 = tf.nn.max_pool(actv2, ksize = [1,pool_kernel_size_layer_2,pool_kernel_size_layer_2,1], strides = [1,pool_strides_layer_2,pool_strides_layer_2,1], padding = 'SAME')

#Defining Variables
layer3 = tf.Variable(tf.random_normal([7*7*layer_2_channels, layer_3_channels]))
bias3 = tf.Variable(tf.random_normal([layer_3_channels]))

#Fully-Connected Layer
reshapef = tf.reshape(pool2, [-1, 7*7*layer_2_channels])
add1 = tf.add(tf.matmul(reshapef, layer3), bias3)
actv3 = tf.nn.relu(add1)
final = tf.nn.dropout(actv3, keep_prob)

#Converting to class scores
layer4 = tf.Variable(tf.random_normal([layer_3_channels, 10]))
bias4 = tf.Variable(tf.random_normal([10]))

output =  tf.add(tf.matmul(final, layer4), bias4)

# Calculating Loss Function
loss_vector = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=training_labels)
cost = tf.reduce_mean(loss_vector)
optimizer = tf.train.AdamOptimizer(learning_rate=rate_learning).minimize(cost)
predicted_number = tf.argmax(output, 1, name = "predicted_number");
correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(training_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#---------------------------------------------------------------------------------------------------------------------------#

init = tf.global_variables_initializer()
saver = tf.train.Saver()
added_training = ComputerMNIST.ComputerMNIST() 
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < num_iterations:
        batch_x = []
        batch_y = []
        # Case 0: Only Use MNIST, Case 1: Use MNIST + Computer MNIST, Case 2: Only Computer MNIST
        if case ==0:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
        elif case ==1:        
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Adding Computer MNIST into batch_x and batch_y
            temp_x, temp_y = added_training.next_batch(ComputerMNIST_batch_size)
            batch_x = np.concatenate((batch_x, temp_x), axis =0)
            batch_y = np.concatenate((batch_y, temp_y), axis =0)
        elif case ==2:
            batch_x, batch_y = added_training.next_batch(ComputerMNIST_batch_size)


        sess.run(optimizer, feed_dict={training_data: batch_x, training_labels: batch_y, keep_prob: dropout_prob})
        
        if(step%100==0):
            loss, acc = sess.run([cost, accuracy], feed_dict={training_data: batch_x, training_labels: batch_y, keep_prob: 1})
            print("Generation " + str(step*batch_size) + " with normal batch, Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
        
    
    ltr = saver.save(sess, "MyModel\\CNNModel")

    data = mnist.test.images[:test_batch_size];
    labels = mnist.test.labels[:test_batch_size];
    
    pred, ac = sess.run([predicted_number, accuracy], feed_dict={training_data: data ,training_labels: labels, keep_prob: 1.})
    
    print("\nModel Predictions:\n")
    print(pred)
    print("\nTest Batch Labels:\n")
    print(np.argmax(labels, 1))
    print("\nTraining Accuracy for Test Batch: " + "{:.5f}".format(ac));