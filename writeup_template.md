#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./results/Simulator.png "Self Driving Car Simulator"
[image2]: ./results/loadimages.png "Load Images"
[image3]: ./results/HSVimage.png "Convert HSV Image"
[image4]: ./results/random_brightness.png "Generated New Bright Images"
[image5]: ./results/fliped_Image.png "Flipped Image"
[image6]: ./results/Cropimage.png "Croped Image (ROI)"
[image7]: ./results/Steering_Angle_Distribution.png "Steering Angle Distribution"
[image8]: ./results/nvidia_network.png "Nvidia Network"
[image9]: ./results/MyModel.png "My Implemented Nvidia Network"
[image10]: ./results/Loss_of_training_and_validation.png "Loss of training and validation"

---

<b>1. Udacity Self-driving-car Simulator based Data Recoding</b>
* The simulator has Training mode and Autonomous mode and two courses. 
* In my case, I only tested on the track1.
* Training mode is used to collect training data by driving through the tracks and recording the driving data. 
* The Autonomous mode is the test mode for our trained model.

![alt text][image1]

---

<b>2. Train Data Sets</b>

To make training model, I used to udacity datasets. Udacity dataset has the number of 24,108 data as below:
* number of images taken from a center camera: 8036
* number of images taken from a camera translated to the left: 8036
* number of images taken from a camera translated to the right: 8036
* color channels: 3 (RGB)
* dimensions: 320x160

Attributes available in the dataset:
* 'center' - center camera image
* 'left' - left camera image
* 'right' - right camera image
* 'steering' - steering angle at the time the images was taken
* 'throttle' - throttle value
* 'brake' - brake value
* 'speed' - speed value

There is an examples of the left, center, and right images.

![alt text][image2]

There is an distribution of the traing steering angle datasets.

![alt text][image7]

To make general training model, I use new datasets using two methods.

* First method is random flipping : I make flipped images of probability 50%.

 ![alt text][image5]
 
* Second method is random brightness of H chanel in the HSV domain : 

 ![alt text][image3]
 ![alt text][image4]

---

<b>3. Traning Model</b>

I use the NVIDIA pipeline for my model because NVIDIA model is the best pipeline for this project in my try and also many students recommend this pipeline. NVIDIA model consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers. I added the cropping layer to NVIDIA model.

![alt text][image8]
![alt text][image9]

First layer is the normalization layer. According to the Nvidia paper, this enables normalization also to be accelerated via GPU processing.

Convolution were used in the first three layers with 2x2 strides and a 5x5 kernel and non-stride convolution with 3x3 kernel size in the last two convolutional layers.

The convolutional layers were followed by three fully connected layers which then outputs the steering angle.

Activation to introduce non-linearities into the model: I chose ELU as my activation.

I used the Dropout (0.5) and L2 Regularization (0.001) on all layer to avoiding the overfitting. This turned out to be a good practice as the model could generalize to the second track, without using it for training.

I used Adam optimizer and learning rate 0.001. I tried to train my model using some learning rate such as 0.01 and 0.005. Finally, I got the best results using learning rate 0.001, so I choose the 0.001 for my model training.

Finally, I got the my trained model: model_hkkim.h5

In this model, I was set ROI area in the images as follows;
![alt text][image6]

* model_hkkim.py : training model
* drive.py : drive a car in a simulator
* model_hkkim.json : saved training model
* model_hkkim.h5 : saved training weight
* Analysis_Model_hkkim.ipynb and Analysis_Model_hkkim.html : Analysis of the training process
* Track1.mp4 : test result of the track1

---

<b>4. Result of Test Simulation</b>

* The trained model was tested on the first track. 
* This gave a good result, as the car could drive on the track smoothly. 
* See Track1.mp4

---

<b>5. Future Works</b>

A deep learning model to drive autonomously on a simulated track was trained by using a human behaviour-sampled data. 
This project is very impressive project. I have looked forward to start this project since I check the curriculum of Udacity's SDC ND. 
During completing this project, I could got used to Keras and python generator. 
I test my model on Udacity second track, but my model couldn't complete driving the course. 
Therefore, I am going to improving make more general model and testing.
After finish the SDC ND, if I have a chance, I want to test it with real vehicle.
