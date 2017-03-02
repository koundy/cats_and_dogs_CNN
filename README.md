# Vgg style Convolutional network for classification

There are many methods for image classification and one of the recent breakthrough method is a convolutional neural networks. I got interested in this aspect of deep learning and want to try and learn coding them myself. This code is meant to help begginers like me to implement a CNN using tensorflow and tflearn. You need to know the theory behind the CNNs and how they work inorder to follow the code.

## Method

Here i used the architecture implemented in the paper Very Deep Convolutional Networks for Large-Scale Image Recognition [http://arxiv.org/pdf/1409.1556]. Except for the following two changes.

1. Reduced the depth of the network by removing last layer of convulutions
2. Used batch normalization technique[https://arxiv.org/abs/1502.03167] in every layer for better performence.

## Results

I have divided the 25k images into 90 % traing set and 10 % validation set. I could obtain 97.4 % accuracy using this code on the validation dataset which is pretty good.
