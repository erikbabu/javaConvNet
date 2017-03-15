# javaConvNet
Convolutional Neural Network implemented in java with option for user to have their own image classified by the network. Some code adapted from https://deeplearning4j.org/mnist-for-beginners

How to use: 
Clone the repository

IntelliJ instructions:
-Open IntelliJ
File -> New -> Project from existing sources
-Select the javaConvNet folder as the project folder 
In the Import Project page, select Import project from external model and select Maven
Click Next
Under select maven projects to import select org.deeplearning4j:dl4j-examples:0.8-SNAPSHOT
Select a java jdk to use. e.g. 1.7 or 1.8 and click next
Click Finish
Open the project (this window or new window)
File -> Project Structure -> Modules
- Navigate to the sources tab. Click on the src folder and click Mark as: Sources, above. 
- Click apply and then click OK
After a few seconds, the src folder should now appear.
Go to the Dependencies tab, click the + button, select import module and click dl4j-examples.iml. Click ok. Click Apply then click ok again. 
If you are getting import errors, right click pom.xml, select maven and click Reimport.
You should now be ready to use the program


PROGRAM INSTRUCTIONS

Stage 1)
a)
You may either use the previously compiled training data (MyCNN.zip) which has been trained using the default values (from when I trained the network) found in LeNetCNN.java. In that case, please skip on to Stage 2. If you would like to train the network yourself, please carry on to stage b). 

b)
Open the LeNetCNN.java file. 
(Optional) change the values of the variables if you would like to alter how the network is trained
Run the file
Wait for about 5 minutes for the training to complete. A message will be displayed showing the accuracy of the network.


Stage 2)
Open the TestYourOwnImage.java file. Run the file. Open the image you would like to test the network on. Preferably enter an image of a single digit as this is what the network is suited to classifying. You can find some sample imamges to try in the img folder. You are more than free to enter an image of an animal to see what type of number the network thinks that a cat looks like. Click ok. After a few seconds, the network should tell you what it number it predicts as well as the output probabilities for each number. If it is not what was expected, use a clearer image or alter the variables in the LeNetCNN.java file until the accuracy becomes higher.

