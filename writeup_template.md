# Traffic Sign Recognition

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/bar_chart.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/augmented.png "Augmented Images"
[image4]: ./examples/web_signs.png "Web Sign Set"
[image5]: ./german_signs/children_crossing.jpg "Traffic Sign 2"
[image6]: ./german_signs/keep_right.jpg "Traffic Sign 3"
[image7]: ./german_signs/no_passing.jpg "Traffic Sign 4"
[image8]: ./german_signs/speed_limit_70.jpg "Traffic Sign 5"
[image9]: ./examples/images_per_class.png "Per Class Example"

## Rubric Points
---
###Writeup / README

Here is a link to my [project code](https://github.com/tawnkramer/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images.
* The size of the validation set is 4410 images.
* The size of test set is 12630 images.
* The shape of a traffic sign image is 32x32 pixels.
* The number of unique classes/labels in the data set is 43 classes.

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing number of sample images per class.

![alt text][image1]

Here are 5 sample images per class, 43 classes.

![alt text][image9]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because this speeds training and in most cases still provides enough detail to distinguish sign types.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because our optimization function of choice, gradient decent, works best when the values are centered on the origin with a mean variance of 1.0.

I decided to generate additional data because image augmentation has been shown to boost generalization in neural networks. 

To add more data to the the data set, I used image transformations that resulted in images less likely, but still possible to be seen in real life. The images were all included twice, once unaltered, and a second time with a random set of variations to brightness, contrast, sharpness, rotation, and shadow overlay.

Here is an example of set of original images paired with their augmented twin:

![alt text][image3]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 5x5 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling			| 2x2 stride, outputs 14x14x6					|
| Convolution 5x5     	| 5x5 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling			| 2x2 stride, outputs 5x5x16					|
| Flatten				| inputs 5x5x16, outputs 400					|
| Fully connected		| inputs 400, outputs 120						|
| RELU					|												|
| Fully connected		| inputs 120, outputs 84						|
| RELU					|												|
| Fully connected		| inputs 84, outputs 10							|
| Softmax				| 1 hot encoding of most likely class			|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer, a variant of stochastic gradient decent which improves performance by calculating momentem to descend the error slope more consistently. Stochastic gradient decent does a gradient calculation with a small sample of data and therefore produces a noisy, inaccurate direction as a tradeoff for speed in calculation. The momentem calculated in the AdamOptimizer helps to smooth this vector with momentum over a number of steps. 

This takes an initial learning rate, as does SGD, but will modify the rate over time. Samples rates between 0.0001 and 0.5 were explored. 0.005 was the final learning rate chosen. This learning rate affects how much of the error differential for a given layer is applied to the weights of the model on each backpropogation step.

Batch size and number of epochs help to tune the machine learning process. The batch size indicates the number of samples that are aggregated when arriving at a single change to apply via backpropogation to the weights of the neural network. Batch sizes between 64 and 256 were considered, with 128 giving optimal results. 

The number of epochs tells the optimizer how many times to run the full loop of prediction, error gradient calculation, and then backwards propogation to move weights in a favorable direction. As a network trains, it can tend to over-fit the data, or memorize particualars of the data in ways that don't extend beyond the test set. The pressure over time to descrease error on the training set can ulitmately hurt it's ablility to generalize. At 30 epochs, we found a reasonable tradeoff between network training time and network performance on the test set.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 94.6%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

LeNet was the suggested architecture, and seemed to fit the problem quite well. It needs the depth to represent the number of features to generalize 43 classes, while not enough weights to make memorization of training set the best choice for error minimization.

* What were some problems with the initial architecture?

The initial architecture worked well. It was time tested and didn't take very much time to validate that it was a fruitful approach. Increasing the weight counts of the final connected layers didn't result in a improvement and only enabled the model to over-fit more easilly.

* How was the architecture adjusted and why was it adjusted?

Sigmoid activation was tested, but quickly discarded as it suffers from poor error propogation and requires many more epochs for similar accuracy. Dropout was tested and didn't significantly impact the result. 

* Which parameters were tuned? How were they adjusted and why?

Batch size, learning rate, and epoch count were all adjusted. Data augmentation was attempted and ulitmately discarded as not aiding to the final result.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

The convolution layer is critical to using the 2d coherency of the underlying structure to help build features that scale to a generalization of the classes. Dropout is normally beneficial to forcing redundancy in weight representation and can act as a normalization presessure on the final solution. In this case dropout didn't result in significant improvements and ultimately the more simple pooling solution worked for this data set.

If a well known architecture was chosen:
* What architecture was chosen?

 The LeNet architecture was chosen.
 
* Why did you believe it would be relevant to the traffic sign application?

The LeNet architecture features enough convolution layers to represent the highly structured sign data set, and enough fully connected layers to tie them together without too much room for over fitting. 

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The final model worked well, and identified 94.5% of the validation images.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are some German traffic signs that I found on the web:

![alt text][image4]

The first image might be difficult to classify because there are many signs with a triangular shape. The second sign is well represented in the test set, and probably not a challenge. The third image of the speed limit was close to the other speed limit signs. The fourth image of the children crossing has a lot of small detail that is mostly lost at this 32x32 resolution. And the no passing sign has close to the fewest samples in the test set. Yield and 70 km/h were included for additinal verification.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General caution      	| General caution   							| 
| Keep right   			| Keep right									|
| Speed limit (60km/h	| Speed limit (60km/h							|
| Children crossing 	| Children crossing 			 				|
| No passing			| No passing     	 							|
| Yield					| Yield     	 								|
| Speed limit (70 km.h)	| Speed limit (70 km.h)							|


The model was able to correctly guess 7 of the 7 traffic signs, which gives an accuracy of 100%. Originally, the test images were cropped too closely and resulted in reduced accuracy of 57%. Once cropping changed to more closely match the test set, the network performed flawlessly on the small sample set.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

The model quite sure of all the images and didn't indicate any hesitantacy.


Predicted sign: General caution, probability: 1.000 correct: True

| Rank | Index | Prob |                       Label                        |
|:----:|:-----:|:----:|:--------------------------------------------------:|
|  1   |   18  | 1.0  |                  General caution                   |
|  2   |   42  | 0.0  | End of no passing by vehicles over 3.5 metric tons |
|  3   |   10  | 0.0  |    No passing for vehicles over 3.5 metric tons    |
|  4   |   17  | 0.0  |                      No entry                      |
|  5   |   16  | 0.0  |      Vehicles over 3.5 metric tons prohibited      |

Predicted sign: Keep right, probability: 1.000 correct: True

| Rank | Index | Prob |                       Label                        |
|:----:|:-----:|:----:|:--------------------------------------------------:|
|  1   |   38  | 1.0  |                     Keep right                     |
|  2   |   42  | 0.0  | End of no passing by vehicles over 3.5 metric tons |
|  3   |   10  | 0.0  |    No passing for vehicles over 3.5 metric tons    |
|  4   |   18  | 0.0  |                  General caution                   |
|  5   |   17  | 0.0  |                      No entry                      |

Predicted sign: Speed limit (60km/h), probability: 1.000 correct: True

| Rank | Index |     Prob    |                    Label                     |
|:----:|:-----:|:-----------:|:--------------------------------------------:|
|  1   |   3   |   0.999998  |             Speed limit (60km/h)             |
|  2   |   5   | 2.17723e 06 |             Speed limit (80km/h)             |
|  3   |   2   | 2.00031e 08 |             Speed limit (50km/h)             |
|  4   |   10  |  1.6189e 24 | No passing for vehicles over 3.5 metric tons |
|  5   |   6   | 3.62349e 28 |         End of speed limit (80km/h)          |

Predicted sign: Children crossing, probability: 1.000 correct: True

| Rank | Index |     Prob    |            Label             |
|:----:|:-----:|:-----------:|:----------------------------:|
|  1   |   28  |     1.0     |      Children crossing       |
|  2   |   8   | 3.22192e 27 |    Speed limit (120km/h)     |
|  3   |   35  | 3.61174e 28 |          Ahead only          |
|  4   |   20  | 6.18503e 31 | Dangerous curve to the right |
|  5   |   12  | 2.04654e 31 |        Priority road         |

Predicted sign: No passing, probability: 1.000 correct: True

| Rank | Index | Prob |                    Label                     |
|:----:|:-----:|:----:|:--------------------------------------------:|
|  1   |   9   | 1.0  |                  No passing                  |
|  2   |   10  | 0.0  | No passing for vehicles over 3.5 metric tons |
|  3   |   18  | 0.0  |               General caution                |
|  4   |   17  | 0.0  |                   No entry                   |
|  5   |   16  | 0.0  |   Vehicles over 3.5 metric tons prohibited   |

Predicted sign: Yield, probability: 1.000 correct: True

| Rank | Index | Prob |                       Label                        |
|:----:|:-----:|:----:|:--------------------------------------------------:|
|  1   |   13  | 1.0  |                       Yield                        |
|  2   |   42  | 0.0  | End of no passing by vehicles over 3.5 metric tons |
|  3   |   10  | 0.0  |    No passing for vehicles over 3.5 metric tons    |
|  4   |   18  | 0.0  |                  General caution                   |
|  5   |   17  | 0.0  |                      No entry                      |

Predicted sign: Speed limit (70km/h), probability: 1.000 correct: True

| Rank | Index |     Prob    |                 Label                 |
|:----:|:-----:|:-----------:|:-------------------------------------:|
|  1   |   4   |     1.0     |          Speed limit (70km/h)         |
|  2   |   1   | 1.36683e 33 |          Speed limit (30km/h)         |
|  3   |   0   |  4.3787e 34 |          Speed limit (20km/h)         |
|  4   |   11  |     0.0     | Right of way at the next intersection |
|  5   |   18  |     0.0     |            General caution            |


