# Getting your hands dirty with deep learning
This repo contains the exercises for the workshop with each exercise and solution in a separate branch.

The complete set of exercises will be in the repo shortly before the workshop.

Before the workshop please complete the intial exercises "ex0-setup" which will ensure all the dependencies and large data files are present on your computer and you'll be ready for the workshop.

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

* You'll also need a large ASCII text file of your choice. This could be the works of Shakespeare, the Linux kernel source (concatenated into a single file) or any text file. We'll use this in one of the exercises to generate text in the same style.
* [OPTIONAL] download some images that you'd like to classify. For example cats vs dogs. If possible you'll want a few hundred of each category

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
