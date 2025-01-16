using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System;
using System.Linq;
using Unity.Mathematics;
using Unity.Burst;
using Unity.Collections;


public delegate float[] ActivationFunction(float[] inputs);
public delegate float[] ActivationDerivative(float[] inputs, float[] prevGradients);
public delegate float ErrorFunction(float[] outputs, float[] expectedOutputs);
public delegate float[] ErrorDerivative(float[] outputs, float[] expectedOutputs);

[Serializable]
public class LayerData
{
    public float[] Weights; // Flattened
    public int WeightsRows; // Number of rows in the original 2D array
    public int WeightsCols; // Number of columns in the original 2D array
    public float[] Biases;  // 1D array, no changes needed
}

[Serializable]
public class NetworkData
{
    public LayerData[] Layers;
}


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
        preActivationOutput = new float[numOutputs];
        postActivationOutput = new float[numOutputs];
        // Xavier initialization
        float initializationRange = 1 / Mathf.Sqrt(numInputs);
        for (int i = 0; i < numOutputs; i++) {
            biases[i] = 0;
            biasGradients[i] = 0;
            for (int j = 0; j < numInputs; j++) {
                weights[i, j] = UnityEngine.Random.Range(-initializationRange, initializationRange);
                weightGradients[i, j] = 0;
            }
        }
    }

    public float[] CalculateOutput(float[] inputs) {
        for (int i = 0; i < numOutputs; i++) {
            preActivationOutput[i] = biases[i];
            for (int j = 0; j < numInputs; j++) {
                preActivationOutput[i] += inputs[j] * weights[i, j];
            }
        }
        postActivationOutput = activationFunction(preActivationOutput);
        return postActivationOutput;
    }
    public void ApplyGradients(float learningRate) {
        for (int i = 0; i < numOutputs; i++) {
            biases[i] -= biasGradients[i] * learningRate;
            biasGradients[i] = 0;
            for (int j = 0; j < numInputs; j++) {
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
        // Create indices array and shuffle it
        int[] indices = new int[inputData.Count];
        for (int i = 0; i < indices.Length; i++) {
            indices[i] = i;
        }
        // Fisher-Yates shuffle
        for (int i = indices.Length - 1; i > 0; i--) {
            int j = UnityEngine.Random.Range(0, i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        float totalCost = 0f;
        int batchCount = 0;
        for (int i = 0; i < inputData.Count; i++) {
            // Use shuffled index to access data
            int idx = indices[i];
            float[] input = inputData[idx];
            float[] output = outputData[idx];
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
                float[] outputGradients = errorDerivative(output, expectedOutput);
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
        return cost/outputs.Length;
    }

    public static float[] MSEDerivative(float[] output, float[] expectedOutput)
    {
        float[] outputGradients = new float[output.Length];
        for (int i = 0; i < output.Length; i++)
        {
            float gradient = 2 / (float)output.Length * (output[i] - expectedOutput[i]);
            outputGradients[i] = gradient;
        }
        return outputGradients;
    }

    public static float CategoricalCrossEntropy(float[] outputs, float[] expectedOutputs)
    {
        float cost = 0f;
        for (int i = 0; i < outputs.Length; i++)
        {
            cost += (-expectedOutputs[i] * Mathf.Log(outputs[i]));
        }
        return cost / outputs.Length;
    }

    public static float[] CategoricalCrossEntropyDerivative(float[] output, float[] expectedOutput)
    {
        float[] outputGradients = new float[output.Length];
        for (int i = 0; i < output.Length; i++)
        {
            float gradient = -expectedOutput[i] / output[i];
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

    public static float[] ReLUDerivative(float[] inputs, float[] prevGradients)
    {
        float[] derivatives = new float[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            derivatives[i] = inputs[i] > 0 ? prevGradients[i] : 0;
        }
        return derivatives;
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
            // (e^a * S - e^a * e^a) / s^2
            // For the derivative with respect to the corresponding input
            float totalDerivative = (exps[i] * (s - exps[i])) / s2 * prevGradients[i];
            for (int j = 0; j < inputs.Length; j++)
            {
                // (e^b / (e^a+e^b+e^c))
                // find wrt da - where a is i and b is j
                // - e^b * (e^a) / S^2
                if (j != i) {
                    totalDerivative += -(exps[i] * exps[j]) / s2 * prevGradients[j];
                }
            }
            derivatives[i] = totalDerivative;
        }
        return derivatives;
    }

    public void ApplyUniqueLayerGradients(float learningRate, int startFromLayer)
    {
        for (int i = startFromLayer; i < layers.Length; i++)
        {
            layers[i].ApplyGradients(learningRate);
        }
    }

    public void SaveNetwork(string path)
    {
        NetworkData data = new NetworkData();
        data.Layers = new LayerData[layers.Length];
        
        for (int i = 0; i < layers.Length; i++)
        {
            LayerData layerData = new LayerData();
            
            // Convert 2D weights array to 1D
            layerData.WeightsRows = layers[i].weights.GetLength(0);
            layerData.WeightsCols = layers[i].weights.GetLength(1);
            layerData.Weights = new float[layerData.WeightsRows * layerData.WeightsCols];
            
            for (int row = 0; row < layerData.WeightsRows; row++)
            {
                for (int col = 0; col < layerData.WeightsCols; col++)
                {
                    layerData.Weights[row * layerData.WeightsCols + col] = layers[i].weights[row, col];
                }
            }
            
            layerData.Biases = layers[i].biases;
            data.Layers[i] = layerData;
        }

        string json = JsonUtility.ToJson(data);
        File.WriteAllText(path, json);
    }

    public void LoadNetwork(string path)
    {
        if (!File.Exists(path))
        {
            Debug.LogError($"No save file found at {path}");
            return;
        }

        string json = File.ReadAllText(path);
        NetworkData data = JsonUtility.FromJson<NetworkData>(json);

        if (data.Layers.Length != layers.Length)
        {
            Debug.LogError("Saved network architecture doesn't match current network architecture");
            return;
        }

        for (int i = 0; i < layers.Length; i++)
        {
            LayerData layerData = data.Layers[i];
            
            // Verify dimensions match
            if (layerData.WeightsRows != layers[i].weights.GetLength(0) || 
                layerData.WeightsCols != layers[i].weights.GetLength(1))
            {
                Debug.LogError($"Layer {i} dimensions don't match saved data");
                return;
            }

            // Convert 1D weights array back to 2D
            for (int row = 0; row < layerData.WeightsRows; row++)
            {
                for (int col = 0; col < layerData.WeightsCols; col++)
                {
                    layers[i].weights[row, col] = layerData.Weights[row * layerData.WeightsCols + col];
                }
            }

            // Copy biases
            Array.Copy(layerData.Biases, layers[i].biases, layerData.Biases.Length);
        }
    }
}