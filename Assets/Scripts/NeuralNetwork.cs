using UnityEngine;
using System.Collections.Generic;

public delegate float[] ActivationFunction(float[] inputs);
public delegate float[] ActivationDerivative(float[] inputs, float[] prevGradients);
public delegate float ErrorFunction(float[] outputs, float[] expectedOutputs);
public delegate float[] ErrorDerivative(float[] outputs, float[] expectedOutputs);

public class Layer {
    private int numInputs;
    private int numOutputs;
    private ActivationFunction activationFunction;

    public float[] preActivationOutput;
    public float[] postActivationOutput;

    public float[,] weights;
    public float[] biases;

    public float[,] weightGradients;
    public float[] biasGradients;

    public Layer(int numInputs, int numOutputs, ActivationFunction activationFunction) {
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.activationFunction = activationFunction;

        weights = new float[numOutputs, numInputs];
        biases = new float[numOutputs];

        weightGradients = new float[numOutputs, numInputs];
        biasGradients = new float[numOutputs];

        // Xavier initialization
        float initializationRange = 1 / Mathf.Sqrt(numInputs);

        for (int i = 0; i < numOutputs; i++) {
            biases[i] = 0;
            biasGradients[i] = 0;
            for (int j = 0; j < numInputs; j++) {
                weights[i, j] = Random.Range(-initializationRange, initializationRange);
                weightGradients[i, j] = 0;
            }
        }
    }

    public float[] CalculateOutput(float[] inputs) {
        float[] outputs = new float[numOutputs];
        for (int i = 0; i < numOutputs; i++) {
            float outputVal = biases[i];
            for (int j = 0; j < numInputs; j++) {
                outputVal += inputs[j] * weights[i, j];
            }
            outputs[i] = outputVal;
        }
        preActivationOutput = outputs;
        postActivationOutput = activationFunction(outputs);
        return postActivationOutput;
    }

    public void ApplyGradients(float learningRate) {
        for (int i = 0; i < numOutputs; i++)
        {
            biases[i] -= biasGradients[i] * learningRate;
            biasGradients[i] = 0;
            for (int j = 0; j < numInputs; j++)
            {
                weights[i, j] -= weightGradients[i, j] * learningRate;
                weightGradients[i, j] = 0;
            }
        }
    }

    public void ResetGradients()
    {
        for (int i = 0; i < numOutputs; i++)
        {
            biasGradients[i] = 0;
            for (int j = 0; j < numInputs; j++)
            {
                weightGradients[i, j] = 0;
            }
        }
    }
}

public class NeuralNetwork
{
    private int[] shape;
    private Layer[] layers;
    private ActivationDerivative outputDerivative;
    private ActivationDerivative activationDerivative;
    private ErrorFunction errorFunction;
    private ErrorDerivative errorDerivative;
    private List<float[]> currentBiasDerivatives;

    public NeuralNetwork(int[] shape, ActivationFunction activationFunction, ActivationFunction outputActivation, ActivationDerivative activationDerivative, ActivationDerivative outputDerivative, ErrorFunction errorFunction, ErrorDerivative errorDerivative) {
        this.shape = shape;

        int numLayers = shape.Length - 1;
        currentBiasDerivatives = new List<float[]>();
        layers = new Layer[numLayers];
        for (int i = 0; i < numLayers-1; i++) {
            layers[i] = new Layer(shape[i], shape[i + 1], activationFunction);
            float[] emptyOutputs = new float[shape[i + 1]];
            currentBiasDerivatives.Add(emptyOutputs);
        }
        layers[numLayers - 1] = new Layer(shape[numLayers - 1], shape[numLayers], outputActivation);
        float[] empty = new float[shape[numLayers]];
        currentBiasDerivatives.Add(empty);

        this.outputDerivative = outputDerivative;
        this.activationDerivative = activationDerivative;
        this.errorDerivative = errorDerivative;
        this.errorFunction = errorFunction;
    }

    public string DebugArray(string startingMessage, float[] arr)
    {
        string str = "[";
        for (int i = 0; i < arr.Length-1; i++)
        {
            str += arr[i].ToString();
            str += ", ";
        }
        str += arr[arr.Length - 1].ToString();
        str += "]";
        return startingMessage + str;
    }

    public string DebugArray(string startingMessage, float[,] arr)
    {
        string str = "[";
        for (int i = 0; i < arr.GetLength(0) - 1; i++)
        {
            string currentBlock = "[";
            for (int j = 0; j < arr.GetLength(1) - 1; j++)
            {
                currentBlock += arr[i, j].ToString();
                currentBlock += ", ";
            }
            currentBlock += arr[i, arr.GetLength(1) - 1].ToString();
            currentBlock += "]";
            str += currentBlock;
            str += ", ";
        }

        string block2 = "[";
        for (int j = 0; j < arr.GetLength(1) - 1; j++)
        {
            block2 += arr[arr.GetLength(0)-1, j].ToString();
            block2 += ", ";
        }
        block2 += arr[arr.GetLength(0)-1, arr.GetLength(1) - 1].ToString();
        block2 += "]";
        str += block2;
        str += "]";

        return startingMessage + str;
    }

    public float[] Evaluate(float[] inputs) {
        // DebugArray("INPUT: ", inputs);
        float[] runningOutput = inputs;
        for (int i = 0; i < layers.Length; i++) {
            runningOutput = layers[i].CalculateOutput(runningOutput);
            // DebugArray("LAYER " + i.ToString() + " OUTPUT: ", runningOutput);
        }
        return runningOutput;
    }

    public float TrainOneEpoch(List<float[]> inputData, List<float[]> outputData, float learningRate, int batchSize) {
        float totalCost = 0f;
        int batchCount = 0;
        for (int i = 0; i < inputData.Count; i++)
        {
            float[] input = inputData[i];
            float[] output = outputData[i];
            totalCost += CalculateGradients(input, output);
            batchCount += 1;
            if (batchCount >= batchSize)
            {
                ApplyGradients(learningRate, batchSize);
            }
        }

        if (batchCount > 0) {
            ApplyGradients(learningRate, batchSize);
        }

        return totalCost / inputData.Count;
    }

    public void ApplyGradients(float learningRate, int batchSize)
    {
        float lr = learningRate / batchSize;
        for (int i = 0; i < layers.Length; i++) {
            layers[i].ApplyGradients(lr);
        }
    }

    public void ResetGradients()
    {
        for (int i = 0; i < layers.Length; i++)
        {
            layers[i].ResetGradients();
        }
    }

    public float CalculateGradients(float[] input, float[] expectedOutput) {
        float[] output = Evaluate(input);
        float cost = errorFunction(output, expectedOutput);

        for (int i = layers.Length - 1; i >= 0; i--)
        {
            if (i == layers.Length - 1)
            {
                float[] outputGradients = MSEDerivative(output, expectedOutput);
                float[] preActivationGradients = outputDerivative(layers[layers.Length - 1].preActivationOutput, outputGradients);
                for (int j = 0; j < shape[i+1]; j++) {
                    layers[i].biasGradients[j] += preActivationGradients[j];
                    currentBiasDerivatives[i][j] = preActivationGradients[j];
                    for (int k = 0; k < shape[i]; k++)
                    {
                        if (i > 0)
                        {
                            layers[i].weightGradients[j, k] += layers[i - 1].postActivationOutput[k] * preActivationGradients[j];
                        } else
                        {
                            layers[i].weightGradients[j, k] += input[k] * preActivationGradients[j];
                        }
                    }
                }
            }

            else
            {
                // Calculate pre+post activation gradients
                float[] postActivationGradients = new float[shape[i+1]];
                for (int j = 0; j < shape[i+1]; j++)
                {
                    float accumulatedGradient = 0f;
                    for (int k = 0; k < shape[i + 2]; k++)
                    {
                        // dC/dZ * dZ/dA
                        // dC/dZ = currentBiasDerivatives[i+1][k]
                        // dZ/dA = layers[i+1].weights[k, j]
                        accumulatedGradient += layers[i+1].weights[k, j] * currentBiasDerivatives[i+1][k];
                    }
                    postActivationGradients[j] = accumulatedGradient;
                }
                float[] preActivationGradients = activationDerivative(layers[i].preActivationOutput, postActivationGradients);

                for (int j = 0; j < shape[i+1]; j++)
                {
                    layers[i].biasGradients[j] += preActivationGradients[j];
                    currentBiasDerivatives[i][j] = preActivationGradients[j];
                    for (int k = 0; k < shape[i]; k++)
                    {
                        if (i > 0)
                        {
                            layers[i].weightGradients[j, k] += layers[i - 1].preActivationOutput[k] * preActivationGradients[j];
                        }
                        else
                        {
                            layers[i].weightGradients[j, k] += input[k] * preActivationGradients[j];
                        }
                    }
                }
            }
        }
        /*
        float h = 0.0001f;
        for (int i = 0; i < layers.Length; i++)
        {
            for (int j = 0; j < shape[i + 1]; j++)
            {
                layers[i].biases[j] += h;
                float[] newOutput = Evaluate(input);
                float newCost = CalculateCost(newOutput, expectedOutput);
                float gradient = (newCost - cost) / h;
                layers[i].biasGradients[j] += gradient;
                layers[i].biases[j] -= h;
                
                for (int k = 0; k < shape[i]; k++)
                {
                    layers[i].weights[j, k] += h;
                    float[] newOutput2 = Evaluate(input);
                    float newCost2 = CalculateCost(newOutput2, expectedOutput);
                    float gradient2 = (newCost2 - cost) / h;
                    layers[i].weightGradients[j, k] += gradient2;
                    layers[i].weights[j, k] -= h;
                }
            }
        }
        */
        return cost;
    }

    public static float MSE(float[] outputs, float[] expectedOutputs)
    {
        float cost = 0f;
        for (int i = 0; i < outputs.Length; i++)
        {
            float dif = outputs[i] - expectedOutputs[i];
            cost += (dif * dif);
        }
        return cost;
    }

    public static float[] MSEDerivative(float[] output, float[] expectedOutput)
    {
        float[] outputGradients = new float[output.Length];
        for (int i = 0; i < output.Length; i++)
        {
            float gradient = 2 * (output[i] - expectedOutput[i]);
            outputGradients[i] = gradient;
        }
        return outputGradients;
    }

    public static float[] ReLU(float[] inputs)
    {
        float[] outputs = new float[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            outputs[i] = Mathf.Max(inputs[i], 0);
        }
        return outputs;
    }

    public static float[] Sigmoid(float[] inputs)
    {
        float[] outputs = new float[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            outputs[i] = 1 / (1 + Mathf.Exp(-inputs[i]));
        }
        return outputs;
    }

    public static float[] SigmoidDerivative(float[] inputs, float[] prevGradients)
    {
        // Helpful to precompute these as opposed to recalculating exp a million times
        float[] derivatives = new float[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            float exp = Mathf.Exp(-inputs[i]);
            derivatives[i] = exp / ((1 + exp) * (1 + exp)) * prevGradients[i];
        }
        return derivatives;
    }

    public static float[] Tanh(float[] inputs)
    {
        float[] outputs = new float[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            float positiveExp = Mathf.Exp(inputs[i]);
            float negativeExp = 1 / positiveExp;
            outputs[i] = (positiveExp - negativeExp) / (positiveExp + negativeExp);
        }
        return outputs;
    }

    public static float[] TanhDerivative(float[] inputs, float[] prevGradients)
    {
        float[] derivatives = new float[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            float positiveExp = Mathf.Exp(inputs[i]);
            float negativeExp = 1 / positiveExp;
            // dA / dZ * dC / dA
            derivatives[i] = 4 / ((positiveExp + negativeExp) * (positiveExp + negativeExp)) * prevGradients[i];
        }
        return derivatives;
    }

    public static float[] Softmax(float[] inputs) {
        float[] outputs = new float[inputs.Length];
        float[] exps = new float[inputs.Length];
        float sum = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            float power = Mathf.Exp(inputs[i]);
            exps[i] = power;
            sum += power;
        }
        for (int i = 0; i < inputs.Length; i++)
        {
            outputs[i] = exps[i] / sum;
        }
        return outputs;
    }

    public static float[] SoftmaxDerivative(float[] inputs, float[] prevGradients) {
        // Helpful to precompute these as opposed to recalculating exp a million times
        float[] exps = new float[inputs.Length];
        float s = 0;
        for (int i = 0; i < inputs.Length; i++) {
            float exp = Mathf.Exp(inputs[i]);
            exps[i] = exp;
            s += exp;
        }
        float s2 = s * s;

        float[] derivatives = new float[inputs.Length];
        for (int i = 0; i < inputs.Length; i++) {
            // For the derivative with respect to the corresponding input
            float totalDerivative = (exps[i] * (s - exps[i])) / s2 * prevGradients[i];
            for (int j = 0; j < inputs.Length; j++)
            {
                if (j != i) {
                    totalDerivative += -(exps[i] * exps[j]) / s2 * prevGradients[j];
                }
            }
            derivatives[i] = totalDerivative;
        }
        return derivatives;
    }
}
