package org.deeplearning4j.examples.convolution;

import org.datavec.image.loader.CifarLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Date;


/**
 * Adapted from the following DL4J example: https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/Cifar.java
 * Original author comment below
 * <p>
 * train model by cifar
 * identification unknown file
 *
 * @author wangfeng
 * @since June 7,2017
 */


public class Cifar {
    protected static final Logger log = LoggerFactory.getLogger(Cifar.class);

    public static final String DATA_VAR = "DEVOXXUK_GHDDL_DATA";
    public static final String DATA_DIR = System.getenv(DATA_VAR);
    private static final String CIFAR_ALEX_NET_MODEL = "trainModelByCifarWithAlexNet_model.zip";

    private static String labelStr = "[]";
    private static int height = 32;
    private static int width = 32;
    private static int channels = 3;
    private static int numLabels = CifarLoader.NUM_LABELS;
    private static int numSamples = 50000;
    private static int batchSize = 100;
    private static int iterations = 1;
    private static int freIterations = 50;
    private static int seed = 123;
    private static boolean preProcessCifar = false;//use Zagoruyko's preprocess for Cifar
    /*
     * To get good performance on this dataset you'll need to train for
     * something like 30-50+ epochs. epochs is set to 1 below to allow
     * you to check that everything seems to be working. If training with a
     * single epoch succeeds try increasing epochs and you should get better
     * results.
     */
    // TODO increase epochs once training for a single epoch succeeds
    private static int epochs = 1;

    public static void main(String[] args) throws Exception {
        Cifar cf = new Cifar();

        try {
            if (args.length > 0) {
                switch (args[0]) {
                    case "train":
                        cf.createAndTrainModel(true);
                        break;
                    case "classify":
                        String[] filenames = new String[args.length - 1];
                        System.arraycopy(args, 1, filenames, 0, filenames.length);
                        cf.classifyImages(filenames);
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

        CifarDataSetIterator cifar = new CifarDataSetIterator(batchSize, numSamples,
                new int[]{height, width, channels}, preProcessCifar, true);
        CifarDataSetIterator cifarEval = new CifarDataSetIterator(batchSize, 10000,
                new int[]{height, width, channels}, preProcessCifar, false);

        labelStr = String.join(",", cifar.getLabels().toArray(new String[cifar.getLabels().size()]));
        log.info("Labels: " + labelStr);

        log.info("Start training (" + epochs + " epochs)\n");
        Date startTraining = new Date();
        for (int i = 0; i < epochs; i++) {
            System.out.println("Epoch=====================" + i);
            model.fit(cifar);
        }
        Date endTraining = new Date();
        long trainingSeconds = (endTraining.getTime() - startTraining.getTime()) / 1000;
        log.info("End training. " + epochs + " in " + trainingSeconds + " seconds");

        log.info("=====eval model========");
        Evaluation eval = new Evaluation(cifarEval.getLabels());
        while (cifarEval.hasNext()) {
            DataSet testDS = cifarEval.next(batchSize);
            INDArray output = model.output(testDS.getFeatureMatrix());
            eval.eval(testDS.getLabels(), output);
        }
        System.out.println(eval.stats());

        saveModel(model, CIFAR_ALEX_NET_MODEL);
    }

    public void classifyImages(String[] filenames) throws Exception {
        MultiLayerNetwork model = loadModel(CIFAR_ALEX_NET_MODEL);
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
                // TODO Add 2 convolution layers with 4x4 kernals, stride of 1x1, padding of 0x0 and convolution mode "same" and 64 filters
                // TODO Use SubsamplingLayer.Builder to create a max pooling layer with a 2x2 cell size
                // TODO Add 2 convolution layers like the first 2 but with 96 output filters
                // TODO Add 2 more convolution layers with a kernel size of 3x3 and 128 output filters
                // TODO Add the final 2 convolution layers with a kernel size of 2x2 and 256 output filters
                // TODO Add another max pooling layer with a cell size of 2x2

                .layer(10, new DenseLayer.Builder().name("ffn1").nOut(1024).learningRate(1e-3).biasInit(1e-3).biasLearningRate(1e-3 * 2).build())
                .layer(11, new DropoutLayer.Builder().name("dropout1").dropOut(0.2).build())
                .layer(12, new DenseLayer.Builder().name("ffn2").nOut(1024).learningRate(1e-2).biasInit(1e-2).biasLearningRate(1e-2 * 2).build())
                .layer(13, new DropoutLayer.Builder().name("dropout2").dropOut(0.2).build())
                .layer(14, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
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
        if (null != DATA_DIR) {
            locationModelFile = new File(DATA_DIR, fileName);
        }
        boolean saveUpdater = false;
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

