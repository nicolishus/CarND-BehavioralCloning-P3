# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: network_summary.PNG "Model Visualization"
[image2]: center_lane.PNG "Center Lane Example"
[image3]: recovery_1.PNG "Recovery Image 1"
[image4]: recovery_2.PNG "Recovery Image 2"
[image5]: recovery_3.PNG "Recovery Image 3"
[image6]: clockwise.PNG "Clockwise Driving"
[image7]: left.jpg "Left Camera"
[image8]: center.jpg "Center Camera"
[image9]: right.jpg "Right Camera"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My CNN consisted of a normalization layer centered at 0, cropping of the image, 5 convolutional
layers, a flatten layer, and 4 fully connected layers with a dropout layer after the first
fully connected layer. This model is based off Nvidia's Self Driving Car network mentioned in the
walkthrough video. A visualization of the network can be seen in a later section.

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py lines 28). I only added one after the largest (and first) convolutional layer as this strategy helped my Traffic Sign Classifier.

I drove clockwise for one lap to vary the data so that it would not overfit. I also added flipped images of all laps along with the side cameras with corrections so it could drive smoother. Recovery data was also added since the network did learn to curve left more than it should.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 84). I used small epochs since the network would stagnate quick according to the valdiation accuracy, and so I could iterate quicker. 'Mse" loss function was used since this was a regression problem rather than a classification problem. 

#### 4. Appropriate training data

Mostly center-driving training data was chosen to keep the vehicle driving on the road. I drove centered for about 95% of the data so the network would replicate this. A clockwise lap was added for better generalization as well. Recovery data was added later.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I first started with LeNet's model with epochs set to 3, 'adam' optimizer, and 'mse' for the loss. It performed badly with the car going off to the side almost right away. So speed things up, I decided to use Nvidia's network as it was designed and tested for this type of problem. I also added the normalization and cropping layers just to expediate things even more. Since the top quarter and and maybe bottom 10% were not really useful for driving, this should have in theory helped a lot. Normalization and centering to the middle was used for the image data to help the 'adam' optimizer since a centering of 0 helps gradient descent converge quicker. This was apparent because my network managed to train quick in few epochs and the change between epochs was not very much according to the validation accuracy. These changes made things dramatically better with the car getting to the first curve before going off.

At this point, the validation loss was actually about 71% and stagnated at three epochs. I used three epochs just to get a feel of what the network was doing and for quicker iterations; I could adjust later if needed. From here, it seemed that the network needed more data. Nvidia's network is very deep and powerful, so using only 2000 image samples I collected of three laps was not enough. I decided to implement a trick shown in class to double the data by flipping the images and multiplying the measurements by negative 1. Doing this and running the network lead to the same validation loss, but it performed better getting passed the first curve. It ran off the road very gently indicating it did not know how to recovery once it was close to a lane marker.

At this point, I decided to add some recovery data by just going on each side of the lanes and recording data on centering from the edges. Doing this made the validation accuracy go down to 66%, but the car performed better and made it to the second curve.

Not having enough data, I added the left and right camera images with a correction value of 20% to triple the already double trainin data. This made my training data size about 27000 images. This dramatically lowered the validation accuracy to 26%, but the model was able to get through the whole course without going off the road. It was actually able to stay centered better than my recoreded training data.

#### 2. Final Model Architecture

The final model architecture (model.py lines 29-42) can be seen in the summary below:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recovery; the network was doing well staying in the ceneter, but could not recover once it was near one of the lane markers. These images show what a recovery looks like starting from the left lane marker and going towards the center:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data set, I drove clockwise as can been seen below:

![alt text][image6]

I also flipped the images and used left and right cameras. Below are what a left, center, and right camera image look like, respectively.

![alt text][image7]

![alt text][image8]

![alt text][image9]

After the collection process, I had about 2000 number of data points. Once I finished preprocessing the data by normalizing, centering at zero, cropping, and adding flipped and right/left images, I had a total of 27000 training samples. 

Finally, I randomly shuffled the data set and put 20% of the data into a validation set. 

I only used the simulator for actual testing since the validation loss and accuracy did not correlate with driving ability since it could not take past information into consideration. The validation loss and accuracy were useful to see if the network was getting better or worse (overfitting).
