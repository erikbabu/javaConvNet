# javaConvNet
Convolutional Neural Network implemented in java with option for user to have their own image classified by the network. Some code adapted from https://deeplearning4j.org/mnist-for-beginners

How to use: Clone the repository

IntelliJ instructions:
-Open IntelliJ
File -> New -> Project from existing sources
-Select the folder containing the files 
In the Import Project page, click import project from external model and select Maven, click next
Click next again
Select a java jdk to use. e.g. 1.7 or 1.8
Open the project (this window or new window)


PROGRAM INSTRUCTIONS

Stage 1)
a)
You may either use the previously compiled training data (MyCNN.zip) which has been trained using the default values found in LeNetCNN.java. In that case, please skip on to Stage 2. If you would like to train the network yourself, please carry on to stage b). 

b)
Open the LeNetCNN.java file. 
(Optional) change the values of the variables if you would like to alter how the network is trained
Run the file
Wait for about 5 minutes for the training to complete. A message will be displayed showing the accuracy of the network.


Stage 2)
Open the TestYourOwnImage.java file. Run the file. Open the image you would like to test the network on. Preferably enter an image of a single digit as this is what the network is suited to classifying. You are more than free to enter an image of an animal to see what type of number the network thinks that a cat looks like. Click ok. After a few seconds, the network should tell you what it predicts the number is as well as the output probabilities. If it is not what was expected, use a clearer image or alter the variables in the LeNetCNN.java file until the accuracy becomes higher.  

