
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[ ]:


# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'data/train.p'
validation_file= 'data/valid.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[ ]:


### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np

print(X_train.shape[0])

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = (X_train.shape[1], X_train.shape[2])

# TODO: How many unique classes/labels there are in the dataset.
n_classes = np.max(y_train) + 1

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

# In[ ]:


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import csv
import random

# Visualizations will be shown in the notebook.
#get_ipython().magic('matplotlib inline')

sig_dict = {}
class_names = []

with open('signnames.csv', 'rt') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        sig_dict[int(row['ClassId'])] = row['SignName']
        class_names.append(row['SignName'])
        

plt.rcdefaults()
fig = plt.figure(figsize=(15,20))
fig.suptitle("Random Sample of Images", fontsize=30)

#sample N random images from data set
for n in range(0, 10):
    i = int(random.random() * n_train)
    img = X_train[i]
    plt.subplot(6,5, n+1) # sets the number of images to show on each row and column
    plt.title(sig_dict[y_train[i]][:20])
    plt.imshow(img)

plt.show()


# In[ ]:


#show a bar graph of class distribution samples
plt.rcdefaults()
fig, ax = plt.subplots()
fig.set_size_inches(10, 15, forward=True)
y_pos = np.arange(n_classes)
class_counts = np.zeros(n_classes)
for i in range(y_train.shape[0]):
    iC = y_train[i]
    class_counts[iC] += 1

ax.barh(y_pos, class_counts, align='center',
        color='green')
ax.set_yticks(y_pos)
ax.set_yticklabels(class_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Count')
ax.set_title('Class distribution over %s samples' % "{:,}".format(n_train), fontsize=30)

plt.show()


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
# 
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[ ]:


from PIL import Image
from PIL import ImageEnhance
import numpy as np
import glob

def augment_image(img, shadow_images=None):
    img = Image.fromarray(img)
    #change the coloration, sharpness, and composite a shadow
    factor = random.uniform(0.5, 2.0)
    img = ImageEnhance.Brightness(img).enhance(factor)
    factor = random.uniform(0.5, 1.0)
    img = ImageEnhance.Contrast(img).enhance(factor)
    factor = random.uniform(0.5, 1.5)
    img = ImageEnhance.Sharpness(img).enhance(factor)
    factor = random.uniform(0.0, 1.0)
    img = ImageEnhance.Color(img).enhance(factor)
    try:
        if shadow_images is not None:
            iShad = random.randrange(0, len(shadow_images))
            shadow = shadow_images[iShad].rotate(random.randrange(-15, 15))
            shadow.thumbnail((64, 64))
            r, g, b, a = shadow.split()
            top = Image.merge("RGB", (r, g, b))
            mask = Image.merge("L", (a,))
            mask = ImageEnhance.Brightness(mask).enhance(random.uniform(0.5, 1.0))
            offset = (random.randrange(-16, 16), random.randrange(-16, 16))
            img.paste(top, offset, mask)
    except:
        #print('failed shadow composite')
        #why does this sometimes fail, but mostly work?
        pass
    return np.asarray(img)

def load_images(mask):
    filenames = glob.glob(mask)
    imgs = []
    for fn in filenames:
        imgs.append(Image.open(fn))
    return imgs


# In[ ]:


plt.rcdefaults()
fig = plt.figure(figsize=(15,20))
fig.suptitle("Randomly Augmented Images", fontsize=30)

shadow_images = load_images('./shadows/*.png')

#sample N random images from data set
for n in range(0, 10):
    i = int(random.random() * n_train)
    img = X_train[i]
    plt.subplot(6,6, n * 2 + 1) # sets the number of images to show on each row and column
    plt.title(sig_dict[y_train[i]][:20])
    plt.imshow(img)
    
    img = augment_image(img, shadow_images)
    plt.subplot(6,6, n * 2 + 2) # sets the number of images to show on each row and column
    plt.title(sig_dict[y_train[i]][:20] + " -aug")
    plt.imshow(img)


plt.show()


# In[ ]:


### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
import sys
import multiprocessing

#3 for RGB, 1 for Greyscale
num_channels = 1 

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.reshape(32, 32, 1) 

def rgb_to_grey_normalized(data_set, num_ch):
    rf = 0.3
    gf = 0.6
    bf = 0.1
    conv = np.array([rf, gf, bf])
    result = np.empty([data_set.shape[0], data_set.shape[1], data_set.shape[2], num_ch])
    for i in range(data_set.shape[0]):
        img = (rgb2gray(data_set[i]) - 128.0) / 128.0
        result[i] = img
    return result

def normalized(data_set, num_ch):
    result = np.empty([data_set.shape[0], data_set.shape[1], data_set.shape[2], num_ch])
    for i in range(data_set.shape[0]):
        img = (data_set[i] - 128.0) / 128.0
        result[i] = img
    return result

def async_aug(new_data, iNewData, origImg, shadows):
    new_data[iNewData] = augment_image(origImg, shadows)
    return 1

def pre_process(data_set, num_ch, label_set):
    shadow_images = load_images('./shadows/*.png')
    
    #double the count of images.
    new_shape = ( data_set.shape[0] * 2, data_set.shape[1], data_set.shape[2], data_set.shape[3])
    new_data_set = np.zeros(new_shape)
    
    new_shape = ( label_set.shape[0] * 2)
    new_label_set = np.zeros(new_shape)
    
    print('pre-processing with augmentation', new_data_set.shape[0], 'images')
    
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    tasks = []
    
    print('preparing tasks')
    #for each image, include the original and the duplicate
    for i in range(data_set.shape[0]):
        
        new_data_set[i * 2] = data_set[i]
        tasks.append((new_data_set, i * 2 + 1, data_set[i], shadow_images, ))
        #new_data_set[i * 2 + 1] = augment_image(data_set[i], shadow_images)
        
        #duplicate lable over two entries
        new_label_set[i * 2] = label_set[i]
        new_label_set[i * 2 + 1] = label_set[i]
        
        if i % 1000 == 0:
            print('.', end="")
    print(' ')
    
    print('launching async tasks')
    results = []
    for t in tasks:
        results.append( pool.apply_async( async_aug, t) )
        
    i = 0
    print('getting results of async tasks')
    for result in results:
        i = i + 1
        result.get()
        if i % 1000 == 0:
            print('.', end="")
        
    print('done.')
        
    if num_ch == 3:
        new_data_set = normalized(new_data_set, num_ch)
    elif num_ch == 1:
        new_data_set = rgb_to_grey_normalized(new_data_set, num_ch)
    return new_data_set, new_label_set


X_trainP, y_trainP = pre_process(X_train, num_channels, y_train)
X_validP, y_validP = pre_process(X_valid, num_channels, y_valid)
X_testP, y_testP = pre_process(X_test, num_channels, y_test)

print(X_trainP[1][1][1])


# In[ ]:


#display a random image. make sure the greyscale and normalize went well
import random
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')

index = random.randint(0, len(X_trainP))
image = X_trainP[index].squeeze()
print(image[0])

plt.figure(figsize=(1,1))
if num_channels == 1:
    plt.imshow(image, cmap="gray")
else:
    plt.imshow(image)
print(y_train[index])
print(X_trainP.shape)


# ### Model Architecture

# In[ ]:


### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from tensorflow.contrib.layers import flatten

def LeNet(x, num_classes, num_ch):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, num_ch, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma), name='conv2_W')
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID', name='conv2') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    #fc0    = tf.nn.dropout(fc0, 0.8)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)
    #fc1    = tf.nn.dropout(fc1, 0.9)
    

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)
    
    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = num_classes.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, num_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(num_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits


# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# In[ ]:


### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
from sklearn.utils import shuffle

X_trainP, y_trainP = shuffle(X_trainP, y_trainP)

EPOCHS = 20
BATCH_SIZE = 128

x = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], num_channels))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

rate = 0.005

print("num_channels", num_channels)

logits = LeNet(x, n_classes, num_channels)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        #batch_x, batch_y = get_augmented_batch(offset, BATCH_SIZE, X_data, y_data)
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


#keep track of and save our best accuracy after each epoch
best_accuracy = 0.0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_trainP)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_trainP, y_trainP = shuffle(X_trainP, y_trainP)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_trainP[offset:end], y_trainP[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validP, y_validP)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            print('saving best model..')
            saver.save(sess, './traffic_signs_lenet')            
            
        print() 


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[ ]:


### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import csv
import glob
from scipy.ndimage import imread
import os

sig_dict = {}

with open('signnames.csv', 'rt') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        sig_dict[int(row['ClassId'])] = row['SignName']
        
print('class 1 is:', sig_dict[1])

image_filenames = glob.glob('./german_signs/*.jpg')

X_user = np.empty([len(image_filenames), image_shape[0], image_shape[1], 3])

#our predictions based on human inspection
y_user = np.array([18, 28, 38, 9, 3, 4, 13], dtype=np.int)
print(y_user.shape)

iImage = 0
plt.figure(figsize=(15,20))
    
for filename in image_filenames:
    image = imread(filename)
    plt.subplot(6,5, iImage+1) # sets the number of images to show on each row and column
    plt.title(os.path.basename(filename))
    plt.imshow(image)
    X_user[iImage] = image
    iImage += 1

X_userP = pre_process(X_user, num_channels)
print('X_userP.shape', X_userP.shape)


# In[ ]:


### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
import prettytable

with tf.Session() as sess:
    #saver.restore(sess, tf.train.latest_checkpoint('.'))
    saver.restore(sess, './traffic_signs_lenet')
    final = tf.nn.softmax(logits=logits)
    output = sess.run(final, feed_dict={x: X_userP, y: y_user})
    #print('output', np.argmax(output, axis=1))
    predictions = np.argmax(output, axis=1)
    confidense = np.max(output, axis=1)
    table = prettytable.PrettyTable(['Sign', 'Confidense', 'Correct'])
    for i in range(len(predictions)):
        p = predictions[i]
        c = confidense[i]
        a = y_user[i]
        v = a == p 
        table.add_row([sig_dict[p], "{:.1f}%".format(c * 100), v])
        #print(sig_dict[p], ", confidense", c, ", correct", v)

    print(table.get_string())
    test_accuracy = evaluate(X_userP, y_user)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


# ### The above images were too tightly cropped. These more loosely cropped images fared better.

# In[ ]:


#loosley cropped images:
image_filenames = glob.glob('./german_signs_loosely_cropped/*.jpg')

X_user = np.empty([len(image_filenames), image_shape[0], image_shape[1], 3])

#our predictions based on human inspection
y_user = np.array([18, 28, 38, 9, 3, 4, 13], dtype=np.int)
print(y_user.shape)

iImage = 0
plt.figure(figsize=(15,20))
    
for filename in image_filenames:
    image = imread(filename)
    plt.subplot(6,5, iImage+1) # sets the number of images to show on each row and column
    plt.title(os.path.basename(filename))
    plt.imshow(image)
    X_user[iImage] = image
    iImage += 1

X_userP = pre_process(X_user, num_channels)
print('X_userP.shape', X_userP.shape)


# ### Predict the Sign Type for Each Image

# In[ ]:


### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
import prettytable

with tf.Session() as sess:
    #saver.restore(sess, tf.train.latest_checkpoint('.'))
    saver.restore(sess, './traffic_signs_lenet')
    final = tf.nn.softmax(logits=logits)
    output = sess.run(final, feed_dict={x: X_userP, y: y_user})
    print('output', np.argmax(output, axis=1))
    predictions = np.argmax(output, axis=1)
    confidense = np.max(output, axis=1)
    table = prettytable.PrettyTable(['Sign', 'Confidense', 'Correct'])
    
    for i in range(len(predictions)):
        p = predictions[i]
        c = confidense[i]
        a = y_user[i]
        v = a == p 
        table.add_row([sig_dict[p],  "{:.1f}%".format(c * 100), v])
        #print(sig_dict[p], ", confidense", c, ", correct", v)

    print(table.get_string())
    test_accuracy = evaluate(X_userP, y_user)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


# ### Analyze Performance

# In[ ]:


### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
with tf.Session() as sess:
    #saver.restore(sess, tf.train.latest_checkpoint('.'))
    saver.restore(sess, './traffic_signs_lenet')

    test_accuracy = evaluate(X_userP, y_user)
    print("Test Accuracy = {:.2f}%".format(test_accuracy * 100))


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tk.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# In[ ]:


### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
import prettytable

predictions = np.argmax(output, axis=1)
probability = np.max(output, axis=1)
K = 5
for i in range(len(predictions)):
    p = predictions[i]
    a = y_user[i]
    q = probability[i]
    c = a == p    
    
    #[-K:] take the last K items
    #[::-1] sort in reverse
    top_5_args = output[i].argsort()[-K:][::-1]
    top_5_prob = output[i][top_5_args]
    top_5_labels = []
    for arg in top_5_args:
        top_5_labels.append(sig_dict[arg])
    
    print('Predicted sign: %s, probability: %0.3f correct: %s' % (sig_dict[p], q, c))
 
    rows = []
    
    #assemble rows of rank, index, probability, and label
    for j in range(K):
        rows.append([j + 1, top_5_args[j], top_5_prob[j], top_5_labels[j]])
    
    #create a text based table with header
    p = prettytable.PrettyTable(['Rank', 'Index', 'Prob', 'Label'])
 
    for row in rows:
        p.add_row(row)
    
    print(p.get_string())
    
    print()

    


# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# ---
# 
# ## Step 4 (Optional): Visualize the Neural Network's State with Test Images
# 
#  This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
# 
#  Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.
# 
# For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
# 
# <figure>
#  <img src="visualize_cnn.png" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above)</p> 
#  </figcaption>
# </figure>
#  <p></p> 
# 

# In[ ]:


### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
            
with tf.Session() as sess:
    saver.restore(sess, './traffic_signs_lenet')
    #saver.restore(sess, tf.train.latest_checkpoint('.'))
    conv2 = None
    for op in sess.graph.get_operations():
        if op.name == 'conv2':
            print('found')
            conv2 = op
            break
    
    print(conv2)
    print(type(conv2))
    print(dir(conv2))
    
    if conv2 is not None:
        outputFeatureMap(X_userP[3], conv2)

