# Realtime Optical Digit Recognition with Deep Learning  [![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-cpu)](https://github.com/michaelzhiluo/OCR-Deep-Learning)
OCR-Deep-Learning uses a webcam projected on a computer screen to identify the digits 0-9. This project uses both MNIST database and my own dataset of computer-digits to train a three-layer Convolutional Neural Network.

The recognition rate for computer digits is around 99.84%, which is a much better improvement than using KNN for recognition (~80%). 

[![Watch the video](https://j.gifs.com/Y6ON7W.gif)](https://www.youtube.com/watch?v=HX0PBi470eY&feature=youtu.be)
## Installation 

For the following files to compile, these modules must be installed.
```shell
$ pip install numpy scipy matplotlib opencv-python tensorflow 
```

## General Overview

### Running Model

* #### TrainMNISTCNN.py
  * This file contains the Tensorflow model and processes the MNIST and my own dataset. The model is saved in OCR-Deep-Learning/MyModel/ folder.

* #### RestoreModel.py
  * This file restores the trained model stored in OCR-Deep-Learning/MyModel and uses OpenCV to process images from webcam and identify the corresponding digits.

### Dataset Generation

The documentation below shows how I produced the dataset for computer digits. The labelled digits are in OCR-Deep-Learning/ComputerMNIST/. There are 500 digits for each class. The MNIST-dataset representation of these labelled digits are located in ComputerMNISTImages.npy and ComputerMNISTLabels.npy.

* #### GenerateCameraImage.py
  * Using frames from a webcam, this program uses OpenCV library to find the digit's region of interest and crops them out. Examples are shown below:

<img src="http://imgur.com/azAph53.jpg" height="200" width="250"><br>
<img src="http://imgur.com/Fv2SrIW.jpg"> <img src="http://imgur.com/GA0d5sd.jpg"> <img src="http://imgur.com/w8x9Dht.jpg"> <img src="http://imgur.com/3D9idJ6.jpg"> <img src="http://imgur.com/Y3GnWjN.jpg"> <img src="http://imgur.com/sseISo5.jpg"> <img src="http://imgur.com/HOZC3ut.jpg"> <img src="http://imgur.com/qDN25pw.jpg"> <img src="http://imgur.com/yfwGEsd.jpg"> <img src="http://imgur.com/nEl3M1J.jpg"> 

* #### ConvertToMNIST.py
  * General API that helps convert a cropped-digit image from the webcame into the MNIST dataset. Contains two important methods. First method uses K=2-Means clustering to remove noise. The second method resizes and whitens the image to 20x? image and pastes it on a black 28x28 background. The image is further sharpened with the PIL library and converted into a 784x1 numpy array like MNIST images.
  
* #### GenerateComputerMNISTFile.py
  * API that uses the labelled digits from OCR-Deep-Learning/ComputerMNIST/ and converts them into a 5000x784 numpy array for images a 5000x10 numpy array for labels. 
  
* #### ComputerMNIST.py
  * A class used in TrainMNISTCNN.py that represents the dataset. Most important method is next_batch which selects a batch of size x from the dataset. 
