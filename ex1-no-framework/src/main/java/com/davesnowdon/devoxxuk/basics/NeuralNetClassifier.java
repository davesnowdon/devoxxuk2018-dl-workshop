package com.davesnowdon.devoxxuk.basics;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Exp;
import org.nd4j.linalg.api.ops.impl.transforms.Log;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.indexing.functions.Value;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

/**
 * A 2 layer neural network (one hidden layer, one output layer) implemented using ND4J
 * <p>
 * When implementing the functions you may find the ND4J user guide helpful https://nd4j.org/userguide
 * <p>
 * Also the INDArray documentation may be useful https://nd4j.org/doc/org/nd4j/linalg/api/ndarray/INDArray.html
 */
public class NeuralNetClassifier {

    INDArray weights1;

    INDArray bias1;

    INDArray weights2;

    INDArray bias2;

    int iteration = 0; // used only for logging cost

    /**
     * Construct neural network and initialize parameters.
     * Weights should be set to small random values.
     *
     * @param inputWidth
     * @param hiddenWidth
     * @param outputWidth
     */
    public NeuralNetClassifier(int inputWidth, int hiddenWidth, int outputWidth) {
        // initialise weights with random values (mean zero, standard deviation 1)
        weights1 = Nd4j.randn(hiddenWidth, inputWidth);
        bias1 = Nd4j.randn(hiddenWidth, 1);
        weights2 = Nd4j.randn(outputWidth, hiddenWidth);
        bias2 = Nd4j.randn(outputWidth, 1);
        iteration = 0;
    }

    public static void main(String[] args) throws Exception {
        double learningRate = 0.001;
        int batchSize = 50;
        int nEpochs = 100;

        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 20;

        final String filenameTrain = new ClassPathResource("/classification/moon_data_train.csv").getFile().getPath();
        final String filenameTest = new ClassPathResource("/classification/moon_data_eval.csv").getFile().getPath();

        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(filenameTrain)));

        // TODO you may find it easier to start with a batch size of 1 and then switch to 50 once your code appears to be working
        //DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, 1, 0, 2);
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 2);


        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(filenameTest)));
        //DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, 1, 0, 2);
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 2);

        NeuralNetClassifier model = new NeuralNetClassifier(numInputs, numHiddenNodes, numOutputs);

        for (int e = 0; e < nEpochs; ++e) {
            System.out.println("Epoch: " + e);
            model.train(trainIter, learningRate);
        }
    }

    /**
     * Train the network for one complete pass through the dataset
     *
     * @param iter
     * @param learningRate
     */
    public void train(DataSetIterator iter, double learningRate) {
        if (!iter.hasNext() && iter.resetSupported()) {
            iter.reset();
        }

        while (iter.hasNext()) {
            DataSet next = iter.next();

            // we want data to be stacked with one column, per input value
            INDArray x = next.getFeatures().transpose();
            INDArray y = next.getLabels().transpose();

            State state = forwardPropagation(x);
            double cost = computeLoss(state.value, y);
            state = backwardPropagation(state, x, y);
            updateParameters(state, learningRate);

            if ((iteration % 10) == 0) {
                System.out.println("Cost after iteration " + iteration + ": " + cost);
            }
            ++iteration;
        }
    }

    /**
     * Given input do one complete forward pass through the network
     * We need to compute the activations for both layer 1 and layer 2.
     * We use z's to denote the weighted sums and a's to denote the activation values
     * a2 is the output of the whole network
     */
    public State forwardPropagation(INDArray input) {
        // z1 = weights1 x input + bias1
        // Hint if we are presenting multiple inputs then add() will fail as the shape will be wrong. Use addColumnVector()
        INDArray z1 = weights1.mmul(input).addColumnVector(bias1);

        // a1 = apply the ReLU activation function to z1
        INDArray a1 = relu(z1);

        // z2 = weights2 x a1 + bias2
        INDArray z2 = weights2.mmul(a1).addColumnVector(bias2);

        // a2 =  apply the softmax activation function to z2
        INDArray a2 = softmax(z2);

        // return output and intermediate values (which we'll use for back propagation)
        State state = new State();
        state.value = a2;
        state.cache.put("z1", z1);
        state.cache.put("a1", a1);
        state.cache.put("z2", z2);
        state.cache.put("a2", a2);
        return state;
    }

    /**
     * Compute the gradients relative to the weights and biases
     * Notation:
     * x = matrix multiplication
     * * = element-wise multiplication
     */
    public State backwardPropagation(State state, INDArray input, INDArray labels) {
        int m = input.columns(); // number of examples

        // get our layer activations from the cache
        INDArray a1 = state.cache.get("a1");
        INDArray a2 = state.cache.get("a2");

        /*
         * The combined derivative of the softmax output and negative log likelihood loss turns out to be simple: A2 - labels
         * For more information see:
         * https://deepnotes.io/softmax-crossentropy
         * https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
         * https://sefiks.com/2017/12/17/a-gentle-introduction-to-cross-entropy-loss-function/
         * https://stats.stackexchange.com/questions/273465/neural-network-softmax-activation
         */
        INDArray dZ2 = a2.sub(labels);

        /*
         * Derivative with respect to layer 2 weights
         * dW2 = dZ2 x a1_T
         * where a1_T = transpose of a1
         */
        INDArray dW2 = dZ2.mmul(a1.transpose()).div(m);

        /*
         * Derivative with respect to layer 2 biases
         * db2 = sum of dZ2 along dimension 1
         */
        INDArray dB2 = dZ2.sum(1).div(m);

        /*
         * Derivative with respect to layer 1 weighted sum (Z1)
         * dZ1 = W2_T x dz2 * g'(z1)
         * where:
         * g'(z1) is the derivative of the activation function for layer 1, which equals reluDerivative(a1) in this case
         */
        INDArray dZ1 = weights2.transpose().mmul(dZ2).mul(reluDerivative(a1));

        /*
         * Derivative with respect to layer 1 weights
         * dW1 = dZ1 x input_T
         */
        INDArray dW1 = dZ1.mmul(input.transpose()).div(m);

        /*
         * Derivative with respect to layer 1 biases
         * dB1 = sum of dZ1 along dimension 1
         */
        INDArray dB1 = dZ1.sum(1).div(m);

        // store the computed gradients for use when we update the network's weights
        state.cache.put("dW1", dW1);
        state.cache.put("dB1", dB1);
        state.cache.put("dW2", dW2);
        state.cache.put("dB2", dB2);
        return state;
    }

    public void updateParameters(State state, double learningRate) {
        // get gradients from state
        INDArray dW1 = state.cache.get("dW1");
        INDArray dB1 = state.cache.get("dB1");
        INDArray dW2 = state.cache.get("dW2");
        INDArray dB2 = state.cache.get("dB2");

        // update each of the parameters by subtracting the gradient scaled by the learning rate
        // Hint can use subi() for in place subtraction
        weights1.subi(dW1.mul(learningRate));
        bias1.subi(dB1.mul(learningRate));
        weights2.subi(dW2.mul(learningRate));
        bias2.subi(dB2.mul(learningRate));
    }

    /**
     * ReLU activation function
     * g(z) = max(0, z)
     * <p>
     * Hint: Take a look at BooleanIndexing
     */
    public INDArray relu(INDArray input) {
        INDArray output = input.dup();
        BooleanIndexing.applyWhere(output, Conditions.lessThan(0), new Value(0));
        return output;
    }

    /**
     * Compute the derivative of the ReLU activation function
     * g'(z) = 0 if z < 0
     * = 1 if z >= 0
     * <p>
     * Hint: Take a look at BooleanIndexing
     */
    public INDArray reluDerivative(INDArray input) {
        INDArray grad = Nd4j.zeros(input.rows(), input.columns());
        INDArray tmp = input.dup();
        BooleanIndexing.applyWhere(tmp, Conditions.greaterThanOrEqual(0), new Value(1));
        BooleanIndexing.assignIf(grad, tmp, Conditions.greaterThanOrEqual(1));
        return grad;
    }

    /**
     * Softmax activation function
     * https://en.wikipedia.org/wiki/Softmax_function
     * For each index i of elements in z, g(z_i) = e^z_i / sum(e^z_j) for all j
     * e^z =  exp(z)
     * <p>
     * Hint: Take a look at the Exp operator and Nd4j.getExecutioner().execAndReturn()
     */
    public INDArray softmax(INDArray input) {
        INDArray exp = Nd4j.getExecutioner().execAndReturn(new Exp(input.dup()));
        double sum = exp.sumNumber().doubleValue();
        INDArray output = exp.div(sum);
        return output;
    }

    /*
     * Compute loss function as negative log likelihood
     * http://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
     */
    public double computeLoss(INDArray actual, INDArray expected) {
        int m = expected.columns(); // number of examples
        INDArray logs = Nd4j.getExecutioner().execAndReturn(new Log(actual.dup()));
        logs.muli(expected);
        double loss = -logs.sumNumber().doubleValue() / m;
        return loss;
    }

    /*
     * If we had two classes and a single output variable we would use this.
     * Normally if we had just 2 output values we would use a single output with 0 or 1 generated by a sigmoid
     * activation. However, DL4J's evaluation functions seem to require multiple outputs so we'll use multi-class cross
     * entropy even though it's not actually required.
     *
    public double computeLoss(INDArray actual, INDArray expected) {
        INDArray logActual = Nd4j.getExecutioner().execAndReturn(new Log(actual.dup()));
        INDArray first = expected.mul(logActual);

        INDArray oneMinusActual = Nd4j.ones(actual.rows(), actual.columns()).sub(actual);
        INDArray oneMinusExpected = Nd4j.ones(expected.rows(), expected.columns()).sub(expected);
        INDArray second = oneMinusExpected.mul(oneMinusActual);

        double loss = -first.add(second).sumNumber().doubleValue();
        return loss;
    }*/

    @Override
    public String toString() {
        return "NeuralNetClassifier{" +
                "weights1=" + weights1 +
                ", bias1=" + bias1 +
                ", weights2=" + weights2 +
                ", bias2=" + bias2 +
                '}';
    }

    public static class State {
        public INDArray value;
        public Map<String, INDArray> cache = new HashMap<>();

        @Override
        public String toString() {
            return "State{" +
                    "value=" + value +
                    ", cache=" + cache +
                    '}';
        }
    }
}
