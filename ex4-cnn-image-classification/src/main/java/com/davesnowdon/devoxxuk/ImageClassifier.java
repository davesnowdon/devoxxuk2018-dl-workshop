package com.davesnowdon.devoxxuk;

import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Random;

/**
 * Slightly modified version of AnimalsClassification.java from https://github.com/deeplearning4j/dl4j-examples
 * <p>
 * Use this to experiment with classifying your own images
 */
public class ImageClassifier {
    protected static final Logger log = LoggerFactory.getLogger(ImageClassifier.class);

    public static final String DATA_VAR = "DEVOXXUK_GHDDL_DATA";
    public static final String DATA_DIR = System.getenv(DATA_VAR);

    private static final String LOGO_ALEX_NET_MODEL = "imageClassifierWithAlexNet_model.zip";

    // TODO set this to the location of your images. Each image class should be in a separate sub-folder
    //private static final String IMAGES_BASE_DIR = "PLEASE_SET_THIS_TO_WHERE_YOU_PUT_YOUR_IMAGES";
    private static final String IMAGES_BASE_DIR = "/home/dns/Documents/presentations/20180305-deep-learning-java/data/cnn/cropped";


    // TODO set the width and height to something appropriate to your dataset
    private static int height = 22;
    private static int width = 75;
    private static int channels = 3;

    // TODO set this to how many image classes you have in your dataset
    private static int numLabels = 2;

    // TODO you will need to set this to the total number of images in your dataset
    private static int numSamples = 3713;

    private static int batchSize = 16;
    private static int iterations = 1;
    private static int freIterations = 50;
    private static int seed = 123;
    protected static Random rng = new Random(seed);
    protected static double splitTrainTest = 0.8;

    // TODO Once you've successfully trained this for a single epoch increase this until you get reasonable results
    private static int epochs = 1;

    public static void main(String[] args) throws Exception {
        if (null == DATA_DIR) {
            System.err.println("Please set the environment variable: " + DATA_VAR + " to the directory where you wish to store data files");
            System.exit(1);
        }

        if ("PLEASE_SET_THIS_TO_WHERE_YOU_PUT_YOUR_IMAGES".equals(IMAGES_BASE_DIR)) {
            System.err.println("Please update your code to set IMAGES_BASE_DIR to the directory in which you placed your images");
            System.exit(1);
        }

        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        ImageClassifier imageClassifier = new ImageClassifier();

        try {
            if (args.length > 0) {
                switch (args[0]) {
                    case "train":
                        imageClassifier.createAndTrainModel(true);
                        break;
                    case "classify":
                        String[] filenames = new String[args.length - 1];
                        System.arraycopy(args, 1, filenames, 0, filenames.length);
                        imageClassifier.classifyImages(filenames);
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
            log.error("Caught error: " + e.toString());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void usage() {
        System.err.println("Usage: train|classify [filename]");
    }

    public void createAndTrainModel(boolean enableUi) throws Exception {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);

        MultiLayerNetwork model = createModel();

        if (enableUi) {
            UIServer uiServer = UIServer.getInstance();
            StatsStorage statsStorage = new InMemoryStatsStorage();
            uiServer.attach(statsStorage);
            model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(freIterations));
        }

        log.info("Load data....");
        /**
         * Data Setup -> organize and limit data file paths:
         *  - mainPath = path to image files
         *  - fileSplit = define basic dataset split with limits on format
         *  - pathFilter = define additional file load filter to limit size and balance batch content
         **/
        PathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // new LogoLabelGenerator();
        File mainPath = new File(IMAGES_BASE_DIR);
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        //BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numSamples, numLabels, 0);
        RandomPathFilter pathFilter = new RandomPathFilter(rng);


        /**
         * Data Setup -> train test split
         *  - inputSplit = define train and test split
         **/
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        /**
         * Data Setup -> transformation
         *  - Transform = how to transform images and generate large dataset to train on
         **/
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);
        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{warpTransform});

        /**
         * Data Setup -> normalization
         *  - how to normalize images and generate large dataset to train on
         **/
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        /**
         * Data Setup -> define how to load data into net:
         *  - recordReader = the reader that loads and converts image data pass in inputSplit to initialize
         *  - dataIter = a generator that only loads one batch at a time into memory to save memory
         *  - trainIter = uses MultipleEpochsIterator to ensure model runs through the data for all epochs
         **/
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator dataIter;
        MultipleEpochsIterator trainIter;

        log.info("Start training (" + epochs + " epochs)\n");
        Date startTraining = new Date();

        for (int i = 0; i < epochs; i++) {
            System.out.println("Epoch=====================" + i);
            // Train without transformations
            recordReader.initialize(trainData, null);
            dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
            scaler.fit(dataIter);
            dataIter.setPreProcessor(scaler);
            model.fit(dataIter);
        }

        Date endTraining = new Date();
        long trainingSeconds = (endTraining.getTime() - startTraining.getTime()) / 1000;
        log.info("End training. " + epochs + " in " + trainingSeconds + " seconds");

        log.info("Evaluate model....");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        Evaluation eval = model.evaluate(dataIter);
        log.info(eval.stats(true));
        System.out.println(eval.stats());

        saveModel(model, LOGO_ALEX_NET_MODEL);
    }

    public void classifyImages(String[] filenames) throws Exception {
        MultiLayerNetwork model = loadModel(LOGO_ALEX_NET_MODEL);
        for (String filename : filenames) {
            classifyImage(model, filename);
        }
    }

    public void classifyImage(MultiLayerNetwork model, String filename) throws Exception {
        File file = new File(filename);
        // Use NativeImageLoader to convert to numerical matrix
        NativeImageLoader loader = new NativeImageLoader(height, width, 3);
        // Get the image into an INDarray
        INDArray image = loader.asMatrix(file);

        /*
         DataNormalization scaler = new ImagePreProcessingScaler(0,1);
         scaler.transform(image);
         */
        long startClassify = System.currentTimeMillis();
        INDArray output = model.output(image);
        String modelResult = output.toString();

        int[] predict = model.predict(image);
        long classifyTimeMs = System.currentTimeMillis() - startClassify;
        modelResult += "===" + Arrays.toString(predict);
        System.out.println(filename + " Result: " + modelResult + " (time " + classifyTimeMs + "ms)");
    }

    public MultiLayerNetwork createModel() throws IOException {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .cacheMode(CacheMode.DEVICE)
                .updater(Updater.ADAM)
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l1(1e-4)
                .regularization(true)
                .l2(5 * 1e-4)
                .list()
                .layer(0, new ConvolutionLayer.Builder(new int[]{4, 4}, new int[]{1, 1}, new int[]{0, 0}).name("cnn1").convolutionMode(ConvolutionMode.Same)
                        .nIn(3).nOut(64).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                        .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2 * 2).build())
                .layer(1, new ConvolutionLayer.Builder(new int[]{4, 4}, new int[]{1, 1}, new int[]{0, 0}).name("cnn2").convolutionMode(ConvolutionMode.Same)
                        .nOut(64).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                        .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2 * 2).build())
                .layer(2, new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2, 2}).name("maxpool2").build())

                .layer(3, new ConvolutionLayer.Builder(new int[]{4, 4}, new int[]{1, 1}, new int[]{0, 0}).name("cnn3").convolutionMode(ConvolutionMode.Same)
                        .nOut(96).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                        .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2 * 2).build())
                .layer(4, new ConvolutionLayer.Builder(new int[]{4, 4}, new int[]{1, 1}, new int[]{0, 0}).name("cnn4").convolutionMode(ConvolutionMode.Same)
                        .nOut(96).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                        .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2 * 2).build())

                .layer(5, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}).name("cnn5").convolutionMode(ConvolutionMode.Same)
                        .nOut(128).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                        .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2 * 2).build())
                .layer(6, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}).name("cnn6").convolutionMode(ConvolutionMode.Same)
                        .nOut(128).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                        .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2 * 2).build())

                .layer(7, new ConvolutionLayer.Builder(new int[]{2, 2}, new int[]{1, 1}, new int[]{0, 0}).name("cnn7").convolutionMode(ConvolutionMode.Same)
                        .nOut(256).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                        .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2 * 2).build())
                .layer(8, new ConvolutionLayer.Builder(new int[]{2, 2}, new int[]{1, 1}, new int[]{0, 0}).name("cnn8").convolutionMode(ConvolutionMode.Same)
                        .nOut(256).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                        .learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2 * 2).build())
                .layer(9, new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2, 2}).name("maxpool8").build())

                .layer(10, new DenseLayer.Builder().name("ffn1").nOut(512).learningRate(1e-3).biasInit(1e-3).biasLearningRate(1e-3 * 2).build())
                .layer(11, new DropoutLayer.Builder().name("dropout1").dropOut(0.2).build())
                .layer(12, new DenseLayer.Builder().name("ffn2").nOut(512).learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2 * 2).build())
                .layer(13, new DropoutLayer.Builder().name("dropout2").dropOut(0.2).build())
                // Could use a signal output neuron with sigmoid activation but 2 outputs with Softmax seems to be preferred by DL4J
                .layer(14, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }


    public MultiLayerNetwork saveModel(MultiLayerNetwork model, String fileName) throws Exception {
        File locationModelFile = new File(fileName);
        boolean saveUpdater = false;
        if (null != DATA_DIR) {
            locationModelFile = new File(DATA_DIR, fileName);
        }
        log.info("Saving model to " + locationModelFile);
        ModelSerializer.writeModel(model, locationModelFile, saveUpdater);
        return model;
    }

    public MultiLayerNetwork loadModel(String fileName) throws Exception {
        File locationModelFile = new File(fileName);
        if (null != DATA_DIR) {
            locationModelFile = new File(DATA_DIR, fileName);
        }
        log.info("Loading model from " + locationModelFile);
        return ModelSerializer.restoreMultiLayerNetwork(locationModelFile);
    }
}
