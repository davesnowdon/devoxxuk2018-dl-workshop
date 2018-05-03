package com.davesnowdon.devoxxuk.basics;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class NeuralNetTests {
    double epsilon = 0.0001;

    @Test
    public void weightsAndBiasesInitializedToCorrectSize() {
        NeuralNetClassifier net = new NeuralNetClassifier(100, 200, 10);
        assertEquals(200, net.weights1.rows());
        assertEquals(100, net.weights1.columns());
        assertEquals(200, net.bias1.rows());
        assertEquals(1, net.bias1.columns());
        assertEquals(10, net.weights2.rows());
        assertEquals(200, net.weights2.columns());
        assertEquals(10, net.bias2.rows());
        assertEquals(1, net.bias2.columns());
    }

    @Test
    public void verifyReLUResults() {
        NeuralNetClassifier net = new NeuralNetClassifier(1, 2, 1);

        double[] testValues = {0.0, 1.0, -0.5, 2.0};
        INDArray input = Nd4j.create(testValues);

        INDArray output = net.relu(input);
        assertEquals(0.0, output.getDouble(0), epsilon);
        assertEquals(1.0, output.getDouble(1), epsilon);
        assertEquals(0.0, output.getDouble(2), epsilon);
        assertEquals(2.0, output.getDouble(3), epsilon);
    }

    @Test
    public void verifyReLUDerivativeResults() {
        NeuralNetClassifier net = new NeuralNetClassifier(1, 2, 1);

        double[] testValues = {0.0, 1.0, -0.5, 2.0};
        INDArray input = Nd4j.create(testValues);

        double[] expectedValues = {1.0, 1.0, 0.0, 1.0};
        INDArray expected = Nd4j.create(expectedValues);

        INDArray output = net.reluDerivative(input);
        System.out.println(output);
        assertTrue(output.equalsWithEps(expected, epsilon));
    }

    @Test
    public void verifySoftmaxResults() {
        NeuralNetClassifier net = new NeuralNetClassifier(1, 1, 1);

        double[] testValues = {1, 2, 3, 4, 1, 2, 3};
        INDArray input = Nd4j.create(testValues);
        double[] expectedValues = {0.02364, 0.06426, 0.17468, 0.47483, 0.02364, 0.06426, 0.17468};
        INDArray expected = Nd4j.create(expectedValues);

        INDArray output = net.softmax(input);

        // output should be a probability distribution than sums to 1
        double sum = output.sumNumber().doubleValue();
        assertEquals(1.0, sum, epsilon);

        assertTrue(output.equalsWithEps(expected, epsilon));
    }

    @Test
    public void testForwardPropagation() {
        NeuralNetClassifier net = networkWithKnownWeightsAndBiases();

        double[] testValues = {1, 2};
        INDArray input = Nd4j.create(testValues).transpose();
        NeuralNetClassifier.State state = net.forwardPropagation(input);

        // output should be a probability distribution than sums to 1
        double sum = state.value.sumNumber().doubleValue();
        assertEquals(1.0, sum, epsilon);

        double[] expectedValues = {0.9983614, 0.0016385565};
        INDArray expected = Nd4j.create(expectedValues).transpose();

        // TODO this is a bit rubbish and relies on almost exact matches
        assertTrue(state.value.equalsWithEps(expected, epsilon));
    }

    /*
     * When the actual output matches the desired out the gradients should be very close to zero
     */
    @Test
    public void testBackwardPropagationWithCorrectResult() {
        NeuralNetClassifier net = networkWithKnownWeightsAndBiases();
        double[] testValues = {1, 2};
        INDArray input = Nd4j.create(testValues).transpose();
        double[] l = {1.0, 0.0};
        INDArray labels = Nd4j.create(l).transpose();

        NeuralNetClassifier.State state = net.forwardPropagation(input);
        state = net.backwardPropagation(state, input, labels);

        INDArray dW1 = state.cache.get("dW1");
        INDArray dB1 = state.cache.get("dB1");
        INDArray dW2 = state.cache.get("dW2");
        INDArray dB2 = state.cache.get("dB2");

        assertTrue(dW1.equalsWithEps(Nd4j.zeros(net.weights1.rows(), net.weights1.columns()), 0.01));
        assertTrue(dB1.equalsWithEps(Nd4j.zeros(net.bias1.rows(), net.bias1.columns()), 0.01));
        assertTrue(dW2.equalsWithEps(Nd4j.zeros(net.weights2.rows(), net.weights2.columns()), 0.01));
        assertTrue(dB2.equalsWithEps(Nd4j.zeros(net.bias2.rows(), net.bias2.columns()), 0.01));
    }

    @Test
    public void testBackwardPropagationWithIncorrectResult() {
        NeuralNetClassifier net = networkWithKnownWeightsAndBiases();
        double[] testValues = {2, 1};
        INDArray input = Nd4j.create(testValues).transpose();
        double[] l = {0.0, 1.0};
        INDArray labels = Nd4j.create(l).transpose();

        NeuralNetClassifier.State state = net.forwardPropagation(input);
        state = net.backwardPropagation(state, input, labels);

        // TODO this test needs to be implemented
    }

    @Test
    public void testUpdateParameters() {
        final double learningRate = 0.1;
        NeuralNetClassifier net = networkWithKnownWeightsAndBiases();

        // random gradients
        INDArray dW1 = Nd4j.randn(net.weights1.rows(), net.weights1.columns());
        INDArray dB1 = Nd4j.randn(net.bias1.rows(), net.bias1.columns());
        INDArray dW2 = Nd4j.randn(net.weights2.rows(), net.weights2.columns());
        INDArray dB2 = Nd4j.randn(net.bias2.rows(), net.bias2.columns());
        NeuralNetClassifier.State state = new NeuralNetClassifier.State();
        state.cache.put("dW1", dW1);
        state.cache.put("dB1", dB1);
        state.cache.put("dW2", dW2);
        state.cache.put("dB2", dB2);

        INDArray expectedWeights1 = net.weights1.sub(dW1.mul(learningRate));
        INDArray expectedBias1 = net.bias1.sub(dB1.mul(learningRate));
        INDArray expectedWeights2 = net.weights2.sub(dW2.mul(learningRate));
        INDArray expectedBias2 = net.bias2.sub(dB2.mul(learningRate));

        net.updateParameters(state, learningRate);

        assertTrue(net.weights1.equalsWithEps(expectedWeights1, 0.01));
        assertTrue(net.bias1.equalsWithEps(expectedBias1, 0.01));
        assertTrue(net.weights2.equalsWithEps(expectedWeights2, 0.01));
        assertTrue(net.bias2.equalsWithEps(expectedBias2, 0.01));
    }

    private NeuralNetClassifier networkWithKnownWeightsAndBiases() {
        NeuralNetClassifier net = new NeuralNetClassifier(2, 4, 2);

        // set weights and biases to known values
        double[][] w1 = {
                {1.28, 0.24},
                {0.80, 1.96},
                {0.26, -1.42},
                {1.33, 1.64}};
        net.weights1 = Nd4j.create(w1);

        double[] b1 = {1.23, -1.96, -0.96, 0.16};
        net.bias1 = Nd4j.create(b1).transpose();

        double[][] w2 = {
                {2.56, 0.50, 0.26, 0.74},
                {-0.30, 1.49, -0.50, 0.45}};
        net.weights2 = Nd4j.create(w2);

        double[] b2 = {0.14, 0.93};
        net.bias2 = Nd4j.create(b2).transpose();

        return net;
    }
}
