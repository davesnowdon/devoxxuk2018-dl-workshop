package org.deeplearning4j.examples.recurrent.word2vecsentiment;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.utilities.DataUtilities;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.net.URL;

/**
 * Example: Given a movie review (raw text), classify that movie review as either positive or negative based on the words it contains.
 * This is done by combining Word2Vec vectors and a recurrent neural network model. Each word in a review is vectorized
 * (using the Word2Vec model) and fed into a recurrent neural network.
 * Training data is the "Large Movie Review Dataset" from http://ai.stanford.edu/~amaas/data/sentiment/
 * This data set contains 25,000 training reviews + 25,000 testing reviews
 * <p>
 * Process:
 * 1. Automatic on first run of example: Download data (movie reviews) + extract
 * 2. Load existing Word2Vec model (for example: Google News word vectors. You will have to download this MANUALLY)
 * 3. Load each each review. Convert words to vectors + reviews to sequences of vectors
 * 4. Train network
 * <p>
 * With the current configuration, gives approx. 83% accuracy after 1 epoch. Better performance may be possible with
 * additional tuning.
 * <p>
 * NOTE / INSTRUCTIONS:
 * You will have to download the Google News word vector model manually. ~1.5GB
 * The Google News vector model available here: https://code.google.com/p/word2vec/
 * Download the GoogleNews-vectors-negative300.bin.gz file
 * Then: set the WORD_VECTORS_PATH field to point to this location.
 *
 * @author Alex Black
 */
public class Word2VecSentimentRNN {
    public static final String DATA_VAR = "DEVOXXUK_GHDDL_DATA";
    public static final String DATA_DIR = System.getenv(DATA_VAR);

    /**
     * Data URL for downloading
     */
    public static final String DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";

    /**
     * Location to save and extract the training/testing data
     */
    public static String DATA_PATH;
    /**
     * Location (local file system) for the Google News vectors. Set this manually.
     */
    public static String WORD_VECTORS_PATH;

    private static final String SENTIMENT_MODEL = "w2vSentiment_model.zip";

    int batchSize = 64;     //Number of examples in each minibatch
    int vectorSize = 300;   //Size of the word vectors. 300 in the Google News model
    // TODO You may want to experiment with traiing for a larger number of epochs once you've tried the network after a single epoch
    int nEpochs = 1;        //Number of epochs (full passes of training data) to train on
    int truncateReviewsToLength = 256;  //Truncate reviews with length (# words) greater than this
    final int seed = 0;     //Seed for reproducibility

    private static final String[] TEST_SENTENCES = {
            // TODO put your own test sentences here
            "Rarely have I seen a film that I enjoyed as much as this one"
    };

    public static void main(String[] args) throws Exception {
        if (null == DATA_DIR) {
            System.err.println("Please set the environment variable: " + DATA_VAR + " to the directory where you wish to store data files");
            System.exit(1);
        }
        DATA_PATH = FilenameUtils.concat(DATA_DIR, "dl4j_w2vSentiment/");
        WORD_VECTORS_PATH = DATA_DIR + "/GoogleNews-vectors-negative300.bin.gz";

        Word2VecSentimentRNN sentimentAnalyser = new Word2VecSentimentRNN();

        try {
            if (args.length > 0) {
                switch (args[0]) {
                    case "train":
                        sentimentAnalyser.createAndTrainModel(true);
                        break;
                    case "analyse":
                        if (args.length > 1) {
                            String[] text = new String[args.length - 1];
                            System.arraycopy(args, 1, text, 0, text.length);
                            sentimentAnalyser.analyse(text);
                        } else {
                            sentimentAnalyser.analyse(TEST_SENTENCES);
                        }
                        break;
                    default:
                        usage();
                        break;
                }
            } else {
                usage();
                System.exit(1);
            }
            System.exit(0);
        } catch (Exception e) {
            System.err.println("Caught error: " + e.toString());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void usage() {
        System.err.println("Usage: train|analyse [text]");
    }

    public void createAndTrainModel(boolean enableUi) throws Exception {
        //Download and extract data
        downloadData();

        Nd4j.getMemoryManager().setAutoGcWindow(10000);  //https://deeplearning4j.org/workspaces

        //Set up network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(Updater.ADAM)  //To configure: .updater(Adam.builder().beta1(0.9).beta2(0.999).build())
                .regularization(true).l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
                .learningRate(2e-2)
                .trainingWorkspaceMode(WorkspaceMode.SEPARATE).inferenceWorkspaceMode(WorkspaceMode.SEPARATE)   //https://deeplearning4j.org/workspaces
                .list()
                .layer(0,
                        // TODO Use GravesLSTM.Builder to create a LSTM layer with vectorSize inputs, 256 outputs and TANH activation
                )
                // TODO Once you've successfully trained the network you might want to experiment with adding additional layers and seeing how that changes performance
                .layer(1,
                        // TODO Use RnnOutputLayer.Builder with SOFTMAX activation, 256 inputs, and 2 outputs to create the output layer
                )
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(10));

        //DataSetIterators for training and testing respectively
        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH));
        SentimentExampleIterator train = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, true);
        SentimentExampleIterator test = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, false);

        System.out.println("Starting training");
        for (int i = 0; i < nEpochs; i++) {
            net.fit(train);
            train.reset();
            System.out.println("Epoch " + i + " complete. Starting evaluation:");

            //Run evaluation. This is on 25k reviews, so can take some time
            Evaluation evaluation = net.evaluate(test);
            System.out.println(evaluation.stats());
        }

        //After training: load a single example and generate predictions
        File firstPositiveReviewFile = new File(FilenameUtils.concat(DATA_PATH, "aclImdb/test/pos/0_10.txt"));
        String firstPositiveReview = FileUtils.readFileToString(firstPositiveReviewFile);
        printExample(net, test, firstPositiveReview);

        File firstNegativeReviewFile = new File(FilenameUtils.concat(DATA_PATH, "aclImdb/test/neg/0_2.txt"));
        String firstNegativeReview = FileUtils.readFileToString(firstNegativeReviewFile);
        printExample(net, test, firstNegativeReview);

        saveModel(net, SENTIMENT_MODEL);
    }

    public void analyse(String[] text) throws Exception {
        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH));
        SentimentExampleIterator test = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, false);

        MultiLayerNetwork net = loadModel(SENTIMENT_MODEL);
        for (String t : text) {
            printExample(net, test, t);
        }
    }

    private void printExample(MultiLayerNetwork net, SentimentExampleIterator test, String text) throws Exception {
        INDArray features = test.loadFeaturesFromString(text, truncateReviewsToLength);
        INDArray networkOutput = net.output(features);
        int timeSeriesLength = networkOutput.size(2);
        INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));

        System.out.println("\n\n-------------------------------");
        System.out.println("Sentence: \n" + text);
        System.out.println("\n\nProbabilities at last time step:");
        System.out.println("p(positive): " + probabilitiesAtLastWord.getDouble(0));
        System.out.println("p(negative): " + probabilitiesAtLastWord.getDouble(1));

        System.out.println("----- Example complete -----");
    }

    public static void downloadData() throws Exception {
        //Create directory if required
        File directory = new File(DATA_PATH);
        if (!directory.exists()) directory.mkdir();

        //Download file:
        String archizePath = DATA_PATH + "aclImdb_v1.tar.gz";
        File archiveFile = new File(archizePath);
        String extractedPath = DATA_PATH + "aclImdb";
        File extractedFile = new File(extractedPath);

        if (!archiveFile.exists()) {
            System.out.println("Starting data download (80MB)...");
            FileUtils.copyURLToFile(new URL(DATA_URL), archiveFile);
            System.out.println("Data (.tar.gz file) downloaded to " + archiveFile.getAbsolutePath());
            //Extract tar.gz file to output directory
            DataUtilities.extractTarGz(archizePath, DATA_PATH);
        } else {
            //Assume if archive (.tar.gz) exists, then data has already been extracted
            System.out.println("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath());
            if (!extractedFile.exists()) {
                //Extract tar.gz file to output directory
                DataUtilities.extractTarGz(archizePath, DATA_PATH);
            } else {
                System.out.println("Data (extracted) already exists at " + extractedFile.getAbsolutePath());
            }
        }
    }

    public static MultiLayerNetwork saveModel(MultiLayerNetwork model, String fileName) throws Exception {
        File locationModelFile = new File(fileName);
        if (null != DATA_DIR) {
            locationModelFile = new File(DATA_DIR, fileName);
        }
        boolean saveUpdater = false;
        System.out.println("Saving model to " + locationModelFile);
        ModelSerializer.writeModel(model, locationModelFile, saveUpdater);
        System.out.println("Model saved");
        return model;
    }

    public MultiLayerNetwork loadModel(String fileName) throws Exception {
        File locationModelFile = new File(fileName);
        if (null != DATA_DIR) {
            locationModelFile = new File(DATA_DIR, fileName);
        }
        System.out.println("Loading model from " + locationModelFile);
        return ModelSerializer.restoreMultiLayerNetwork(locationModelFile);
    }
}
