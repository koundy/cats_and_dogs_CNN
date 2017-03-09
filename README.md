# Vgg style Convolutional network for classification

There are many methods for image classification and one of the recent breakthrough method is a convolutional neural networks. I got interested in this aspect of deep learning and want to try and learn coding them myself. This code is meant to help begginers like me to implement a CNN using tensorflow and tflearn. You need to know the theory behind the CNNs and how they work inorder to follow the code.

## Cats and dogs

I have used cats and dogs dataset from kaggle [https://www.kaggle.com/c/dogs-vs-cats/data] to practice.

## Method

Here i used the architecture implemented in the paper Very Deep Convolutional Networks for Large-Scale Image Recognition [http://arxiv.org/pdf/1409.1556]. Except for the following two changes.

1. Reduced the depth of the network by removing last layer of convolutions.
2. Used batch normalization technique[https://arxiv.org/abs/1502.03167] in every layer for better performence.

## Results

I have divided the 25k images into 90 % traing set and 10 % validation set. I could obtain 97.4 % accuracy using this code on the validation dataset which is pretty good.

![vgg_cats_and_dogs](https://cloud.githubusercontent.com/assets/4981114/23499303/9a957666-ff51-11e6-8ea2-c3a3f5d71f7a.png)

## Code

Just download the data and run the above code to see it working. You have to give the path to the directory in which images are situated. It took 50 epochs to converge and about 5 - 6 hours of computation time on Nvidia GTX 1080 GPU with 8 gb memorydshgjkhiueswhfghjksh
