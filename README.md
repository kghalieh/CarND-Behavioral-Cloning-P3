# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview
---
This repository contains my solution for the Udacity Self-Driving Car Engineer Nanodegree Behavioral Cloning project.

The goal of this project is to build and train a neural network that can clone driving behavior. The newtwork is implemented using Keras and trained and validated on data created by Udacity simulator. The simulator enables you to steer a car around  can steer a car around a track for data collection. The data of interest in our case are the images taken by the center camera in the simulated car and its steering angle. 


The project will have five main files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car)
* model.h5 (a trained Keras model)
* a report writeup file (README.md)
* video.mp4 (a video recording of the vehicle driving autonomously around the track for one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.


The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)  

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.  [ In my case becuase I already have CUDA 10 it was better to look at the environment-gpu.yml file and install the missing packages] 

Note that I am using Keras 2.3.1, you should have the same Keras version to use the provided model in the simulator or just start the training from scratch with whatever 1kers version you have. 

The simulator can be downloaded from [here](https://github.com/udacity/self-driving-car-sim).

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 out
```

The fourth argument, `out`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls out

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py out
```

Creates a video based on images found in the `out` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `out.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py out --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

### Very important Tips from the master repo 
" Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles."

I missed this tip and had a hard time figuring out why my models are not doing well, once I read it again I fixed it by adding this line of code to drive.py at line 66
```
image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
```

### Approach

#### 1. Model architecture 

I used the same nVidia model suggested in the project video lessons, the figure below describes  model architecture.


![nVidia model](./report-media/nVidia_model.png)



I only had to change the input shape to 160x 320x 3 and add the cropping layer to only include the road part as much as possible, this will help in reducing the required computation by removing unwanted data.

The model implementations can be found in _[model.py Lines:108- 125]_ 

#### 2. Reduce overfitting in the model

The used model does not include any dropout layers since it did will and not overfit. other used method to avoid overfitting include data augmentation to produce 8 images for each record in the driving log _[model.py lines:52-98]_ and data shuffling. 

#### 4. Training data
Data collection using the simulator included going around the track in both directions for multiple rounds then adding recovery data to help the model in deal with cases like:
- Too close to the edge.
- No curb side.
- get away from bridge corners.
- recover from sharp turn toward the curb.

#### 5. Training 

The model summary as shown by Keras is as follows:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 60, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 28, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 4, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 2, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 34, 64)         16448     
_________________________________________________________________
flatten_1 (Flatten)          (None, 2176)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               217700    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 334,139
Trainable params: 334,139
Non-trainable params: 0
_________________________________________________________________
```

In order to pass the data to the model training process, a generator is used. Generator are a powerful technique to handle large amount of data in patches without the need overload the resources. All the data augmentation can be done "on the go" and sent to the model, once they are used they don't have to be stored in memory. The generator implementation can be found in _[ model.py lines:40-105]_

I used Adam optimizer with the default settings; small learning rate requires more number of epochs wgich was not necessary. The model was fine using only five epochs.


### Result

The model is able to predict the steering angle required to pass the turn for track 1, it can drive well fr track 2 but requires more training to pass it, note that track 2  is unseen by the model

A video demo is [here](https://youtu.be/aCyWo7mlSos)

### Conclusion

This was a great experiment, I had a hard time trying more and more data and models because I mmissed that the drive.py and model.py see different images format PIL and CV2. After that I can say you don't need a lot of data to make this model pass track 1, but it's necessary to generalize more and be able to pass different tracks
