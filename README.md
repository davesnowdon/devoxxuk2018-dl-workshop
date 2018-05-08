# Getting your hands dirty with deep learning
This repo contains the exercises for the workshop with each exercise and solution in a separate branch.

The complete set of exercises will be in the repo shortly before the workshop.

Before the workshop please complete the initial exercise "EX0" which will ensure all the dependencies and large data files are present on your computer and you'll be ready for the workshop.

## EX0 setup

Please complete the following steps before the workshop:

* Install a Java 8 JDK 
    * note: DL4J does not currently work with java9 and above)
 	* Oracle JDK
        * http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html
    * Other JDK sources
        * Azul
            * https://www.azul.com/downloads/zulu/
        * OpenJDK
            * https://adoptopenjdk.net/?variant=openjdk8 (select language from bottom right corner) 

* Set the environment variable `DEVOXXUK_GHDDL_DATA` to the directory where you would like data files to be placed
    * Linux / MacOSX
        * `export DEVOXXUK_GHDDL_DATA=/my/data/directory`
    * Windows
        * `set DEVOXXUK_GHDDL_DATA=c:\my\data\directory`
* Download the `Google word2vec mappings` from https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz and place the file `GoogleNews-vectors-negative300.bin.gz` in the directory indicated by `DEVOXXUK_GHDDL_DATA`
    * Using: 
        * `wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"`
    * SHA1 hash: `be9bfda5fdc3b3cc7376e652fc489350fe9ff863`

* [OPTIONAL] Download some images that you'd like to classify. For example cats vs dogs. If possible you'll want a few hundred of each category. The images don't need to be large as you'll likely want to configure the network to scale them to be less than 100x100 anyway. PyImageSearch has some tips on building up a collection of images using Google image search here: https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/ It's up to you how many different classes of image you want but each class should go in a separate sub-folder.

* [OPTIONAL] You'll also need a large ASCII text file of your choice. This could be the works of Shakespeare, the Linux kernel source (concatenated into a single file) or any text file. We'll use this in one of the exercises to generate text in the same style. For example something like https://s3.amazonaws.com/dl4j-distribution/pg100.txt

* Run 
    * Linux/MacOSX:
        * `./gradlew :ex0-setup:ex0run`
    * Windows: 
        * `gradlew.bat :ex0-setup:ex0run`

* The above `gradle` command does the following:
    * downloads java dependencies
    * checks the environment variable is set
    * checks `GoogleNews-vectors-negative300.bin.gz` is in the right place
    * downloads the `Stanford movie reviews` dataset
    * downloads the `CIFAR10` dataset
    
* Running the `gradle` command may generate some warnings but the last thing printed should be "All good" 


## EX1 Basic neural network without using a framework
In this exercise we'll develop a neural network from first principles using the ND4J library.
ND4J (https://nd4j.org) is the java equivalent of the python numpy library (http://www.numpy.org/)

Check out the exercise branch and address the TODOs to make the unit tests pass. You should then be able to run the following gradle command to train the network:

* Linux/MacOSX:
    * `./gradlew :ex1-no-framework:ex1run`
* Windows: 
    * `gradlew.bat :ex1-no-framework:ex1run`

While doing this exercise you may find it helpful to refer to the ND4J user guide: https://nd4j.org/userguide

Branches:
* Exercise: ex1
* Solution: ex1-solution


## EX2 Basic neural network using DL4J
This exercises uses the same data as for exercise 1 but instead of implementing the neural network ourselves we will use DL4J to define and train the network.

Check out the exercise branch and address the TODOs. You should then be able to run the following gradle command to train the network:

* Linux/MacOSX:
    * `./gradlew :ex2-dl4j:ex2run`
* Windows: 
    * `gradlew.bat :ex2-dl4j:ex2run`

Branches:
* Exercise: ex2
* Solution: ex2-solution


## EX3 Image classification using a CNN
Convolutional Neural Networks (CNNs) are the reason why deep learning has been able to obtain human comparable results for tasks such as image classification. 
As well as image processing, CNNs have been used for speech recognition and some Natural Language Processing (NLP) tasks.

In this exercise you will use DL4J to construct a neural network capable of classifying images from the CIFAR10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html).

Check out the exercise branch and address the TODOs. You should then be able to run the following gradle command to train the network:

* Linux/MacOSX:
    * `./gradlew :ex3-cnn-cifar10:train`
* Windows: 
    * `gradlew.bat :ex3-cnn-cifar10:train`

When training is complete the model parameters will be saved in the directory specified by the environment variable `DEVOXXUK_GHDDL_DATA`
You can then run the classify task to see how well the model performs with some example images. Feel free to modify the build.gradle file to change the images to ones of your choice.

* Linux/MacOSX:
    * `./gradlew :ex3-cnn-cifar10:classify`
* Windows: 
    * `gradlew.bat :ex3-cnn-cifar10:classify`

Branches:
* Exercise: ex3
* Solution: ex3-solution


## EX4 Image classification using a CNN and a custom dataset (optional)
In this exercise we will train a CNN on images of your choice. 

The code changes are trivial (and marked with TODO) and are simply there to adjust the model to the characteristics of your dataset.

You may find the following PyImageSearch blog post helpful when assembling your dataset: https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/

Train
* Linux/MacOSX:
    * `./gradlew :ex4-cnn-image-classification:train`
* Windows: 
    * `gradlew.bat :ex4-cnn-image-classification:train`
    
Classify
* Linux/MacOSX:
    * `./gradlew :ex4-cnn-image-classification:classify`
* Windows: 
    * `gradlew.bat :ex4-cnn-image-classification:classify`

Branches:
* Exercise: ex4 - there is no solution branch for this exercise since only trivial changes to the code are required


## EX5 Sentiment classification using an RNN
Previously we've looked at fixed size data both numerical (ex1 & ex2) and images (ex3 & ex4).
However we often need to process data which does not have a fixed length - a common example of this is natural language text.

Recurrent Neural Networks (RNNs) use not only the current input but also state from previous items in a sequence to allow them to operate on arbitrary length sequences.

When processing text we can either process word-by-word or character-by-character. In this exercise we will process text as words and in exercise 6 we will use characters.

In order to provide the network with semantic information about the words in the text we will use a pre-trained word2vec "embedding." 
Wikipedia has an article describing word2vec here: https://en.wikipedia.org/wiki/Word2vec
Word2vec is just one example of an embedding, there are other models such as GloVe.

Check out the exercise branch and address the TODOs. You should then be able to run the following gradle commands to train the network and analyse text:
  
Train
* Linux/MacOSX:
    * `./gradlew :ex5-rnn-word2vecsentiment:train`
* Windows: 
    * `gradlew.bat :ex5-rnn-word2vecsentiment:train`
    
Analyse
* Linux/MacOSX:
    * `./gradlew :ex5-rnn-word2vecsentiment:analyse`
* Windows: 
    * `gradlew.bat :ex5-rnn-word2vecsentiment:analyse`

Branches:
* Exercise: ex5
* Solution: ex5-solution


## EX6 Generate test using a RNN (optional)
Exercise 5 used a RNN to classify the sentiment of supplied text. 
However, RNNs can also be used to generate text (or any other sequential data). 
This exercise has only trivial code changes and allows you to train a RNN to generate text based on a file of your choice.
For example you could use source code or literature as the basis for training the RNN.

An example of a file you could use to train the network is the complete works of Shakespeare: https://s3.amazonaws.com/dl4j-distribution/pg100.txt 

Train the network and generate text
* Linux/MacOSX:
    * `./gradlew :ex6-rnn-text-generation:train`
* Windows: 
    * `gradlew.bat :ex6-rnn-text-generation:train`

Generate text with pre-trained model
* Linux/MacOSX:
    * `./gradlew :ex6-rnn-text-generation:generate`
* Windows: 
    * `gradlew.bat :ex6-rnn-text-generation:generate`    

Branches:
* Exercise: ex6 - there is no solution branch for this exercise since only trivial changes to the code are required.


## Running with GPU acceleration
If you have a recent NVIDIA GPU you should be able to get improved performance by enabling the CUDA backend for ND4J.

Install CUDA 8 (CUDA 9 is not yet supported) using the instructions on NVIDIA's site: https://developer.nvidia.com/cuda-80-ga2-download-archive

Once you have installed CUDA you will need to install CUDNN, NVIDIA's library, of accelerated code for deep neural networks. Details on NVIDIA's site: https://developer.nvidia.com/cudnn 
In order to download CUDNN you'll need to register for the NVIDIA Developer Program (this is free of charge).

In the dependencies section of the top-level build.gradle you'll see the following lines:

    // use the following to use CPU
    compile "org.nd4j:nd4j-native-platform:${dl4j_version}"
    // or the following to use GPU (requires CUDA8 already installed)
    //compile "org.nd4j:nd4j-cuda-8.0-platform:${dl4j_version}"

Comment the native platform dependency and uncomment the CUDA dependency to give this:

    // use the following to use CPU
    //compile "org.nd4j:nd4j-native-platform:${dl4j_version}"
    // or the following to use GPU (requires CUDA8 already installed)
    compile "org.nd4j:nd4j-cuda-8.0-platform:${dl4j_version}"

## Running on an AWS EC2 instance

__NOTE: This does not seem to work with current AWS deep learning AMIs that support cuda 8 & 9__

You can run the code on this repo on an AWS GPU instance:
* Select a GPU instance such as a p3.2xlarge
* Select the following AMI "Deep Learning AMI (Ubuntu)" this has CUDA and CUDNN pre-installed.
* You'll then want to enable the CUDA backend for ND4J, see above.
* If you want to use the Oracle JDK you'll need to edit your PATH to remove /home/ubuntu/anaconda3/bin

## Running in offline mode
@neomatrix369 added the ability to package this repo with the depedencies and run in offline mode.

Install offline plugin:
`gradle updateOfflineRepository -PofflineRepositoryRoot=./offline-repository`

Download dependencies into offline-repository: 
`gradle updateOfflineRepository -PofflineRepositoryRoot=./offline-repository :ex0-setup:ex0run`

Run task in offline mode:
`gradle -PofflineRepositoryRoot=./offline-repository :ex0-setup:ex0run --offline`

## Thanks
Thanks to Skymind (https://skymind.ai/) who maintain DL4J and ND4J

Exercises 2-6 are taken (with minor modifications) from the DL4J examples repo: https://github.com/deeplearning4j/dl4j-examples