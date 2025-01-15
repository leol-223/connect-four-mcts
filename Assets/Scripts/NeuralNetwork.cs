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
        
        preActivationOutput = new float[numOutputs];
        postActivationOutput = new float[numOutputs];

        weightGradients = new float[numOutputs, numInputs];
        biasGradients = new float[numOutputs];

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

    [BurstCompile]
    public float[] CalculateOutput(float[] inputs) {
        // Preallocate arrays as class fields instead of creating new ones each time
        for (int i = 0; i < numOutputs; i++) {
        float sum = biases[i];
        int vectorSize = 4;
        int vectorCount = numInputs / vectorSize;
        
        for (int j = 0; j < vectorCount * vectorSize; j += vectorSize) {
                float4 inputVec = new float4(
                    inputs[j], inputs[j + 1], 
                    inputs[j + 2], inputs[j + 3]
                );
                float4 weightVec = new float4(
                    weights[i, j], weights[i, j + 1],
                    weights[i, j + 2], weights[i, j + 3]
                );
                sum += math.dot(inputVec, weightVec);
            }
            
            // Handle remaining elements
            for (int j = vectorCount * vectorSize; j < numInputs; j++) {
                sum += inputs[j] * weights[i, j];
            }
            
            preActivationOutput[i] = sum;
        }
        postActivationOutput = activationFunction(preActivationOutput);
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

public class SharedNeuralNetworks
{
    private int[] sharedShape;
    private int sharedLayerCount;
    public Layer[] sharedLayers;
    public NeuralNetwork network1;
    public NeuralNetwork network2;
    
    public SharedNeuralNetworks(
        int[] sharedShape, 
        int[] network1Shape, 
        int[] network2Shape,
        ActivationFunction sharedActivation,
        ActivationFunction network1OutputActivation,
        ActivationFunction network2OutputActivation,
        ActivationDerivative sharedDerivative,
        ActivationDerivative network1OutputDerivative,
        ActivationDerivative network2OutputDerivative,
        ErrorFunction network1ErrorFunction,
        ErrorFunction network2ErrorFunction,
        ErrorDerivative network1ErrorDerivative,
        ErrorDerivative network2ErrorDerivative)
    {
        this.sharedShape = sharedShape;
        this.sharedLayerCount = sharedShape.Length - 1;
        
        Debug.Log($"Creating shared layers: {string.Join(",", sharedShape)}");
        Debug.Log($"Creating value network: {string.Join(",", network1Shape)}");
        Debug.Log($"Creating policy network: {string.Join(",", network2Shape)}");
        
        // Create shared layers
        sharedLayers = new Layer[sharedLayerCount];
        for (int i = 0; i < sharedLayerCount; i++)
        {
            sharedLayers[i] = new Layer(sharedShape[i], sharedShape[i + 1], sharedActivation);
        }

        // Verify shapes match at connection points
        if (network1Shape[0] != sharedShape[sharedShape.Length - 1] ||
            network2Shape[0] != sharedShape[sharedShape.Length - 1])
        {
            Debug.LogError("Network shapes don't match at shared layer connection point");
        }

        // Create individual networks with shared layers
        network1 = new NeuralNetwork(
            network1Shape, // Pass the complete shape
            sharedActivation, 
            network1OutputActivation,
            sharedDerivative,
            network1OutputDerivative,
            network1ErrorFunction,
            network1ErrorDerivative,
            sharedLayers);

        network2 = new NeuralNetwork(
            network2Shape, // Pass the complete shape
            sharedActivation,
            network2OutputActivation,
            sharedDerivative,
            network2OutputDerivative,
            network2ErrorFunction,
            network2ErrorDerivative,
            sharedLayers);
    }

    public float TrainOneEpoch(
        List<float[]> inputData, 
        List<float[]> outputData1,
        List<float[]> outputData2,
        float learningRate,
        int batchSize)
    {
        float totalCost = 0f;
        int batchCount = 0;

        for (int i = 0; i < inputData.Count; i++)
        {
            float[] input = inputData[i];
            float[] output1 = outputData1[i];
            float[] output2 = outputData2[i];

            // Calculate gradients for both networks
            float cost1 = network1.CalculateGradients(input, output1);
            float cost2 = network2.CalculateGradients(input, output2);
            totalCost += (cost1 + cost2) / 2;

            batchCount++;
            if (batchCount >= batchSize)
            {
                ApplyGradients(learningRate, batchSize);
                batchCount = 0;
            }
        }

        if (batchCount > 0)
        {
            ApplyGradients(learningRate, batchSize);
        }

        return totalCost / inputData.Count;
    }

    private void ApplyGradients(float learningRate, int batchSize)
    {
        float lr = learningRate / batchSize;
        
        // Apply gradients to shared layers
        for (int i = 0; i < sharedLayerCount; i++)
        {
            sharedLayers[i].ApplyGradients(lr);
        }

        // Apply gradients to unique layers of each network
        network1.ApplyUniqueLayerGradients(lr, sharedLayerCount);
        network2.ApplyUniqueLayerGradients(lr, sharedLayerCount);
    }

    public void SaveNetworks(string valueFilePath, string policyFilePath)
    {
        // Save value network (network1) with shared layers
        Debug.Log("Saving networks with shared layer count: " + sharedLayerCount);
        network1.SaveNetwork(valueFilePath, 0, 0);  // Save all layers including shared

        // Save policy network (network2) without shared layers
        network2.SaveNetwork(policyFilePath, sharedLayerCount, sharedLayerCount);  // Only save unique layers
    }

    public void LoadNetworks(string valueFilePath, string policyFilePath)
    {
        // Load value network (network1) completely
        network1.LoadNetwork(valueFilePath, 0, 0);

        // Update shared layers reference
        for (int i = 0; i < sharedLayerCount; i++)
        {
            sharedLayers[i] = network1.layers[i];
        }

        // Load policy network (network2) unique layers only
        network2.LoadNetwork(policyFilePath, sharedLayerCount, sharedLayerCount);

        // Update policy network shared layers
        for (int i = 0; i < sharedLayerCount; i++)
        {
            network2.layers[i] = sharedLayers[i];
        }
    }

    public float[] GetValuePrediction(float[] input)
    {
        return network1.Evaluate(input);
    }

    public float[] GetPolicyPrediction(float[] input)
    {
        return network2.Evaluate(input);
    }
}

public class NeuralNetwork
{
    private int[] shape;
    public Layer[] layers;
    private ActivationDerivative outputDerivative;
    private ActivationDerivative activationDerivative;
    private ErrorFunction errorFunction;
    private ErrorDerivative errorDerivative;
    private List<float[]> currentBiasDerivatives;

    public NeuralNetwork(
        int[] shape, 
        ActivationFunction activationFunction, 
        ActivationFunction outputActivation,
        ActivationDerivative activationDerivative,
        ActivationDerivative outputDerivative,
        ErrorFunction errorFunction,
        ErrorDerivative errorDerivative,
        Layer[] sharedLayers = null)
    {
        // Calculate the complete shape including shared layers
        if (sharedLayers != null)
        {
            // Create a new shape array that includes both shared and unique layers
            int[] completeShape = new int[shape.Length + sharedLayers.Length];
            
            // First layer input size
            completeShape[0] = sharedLayers[0].weights.GetLength(1);
            
            // Add shared layer sizes
            for (int i = 0; i < sharedLayers.Length; i++)
            {
                completeShape[i + 1] = sharedLayers[i].weights.GetLength(0);
            }
            
            // Add unique layer sizes
            for (int i = 1; i < shape.Length; i++)
            {
                completeShape[i + sharedLayers.Length] = shape[i];
            }
            
            this.shape = completeShape;
        }
        else
        {
            this.shape = shape;
        }

        int numLayers = shape.Length - 1;
        currentBiasDerivatives = new List<float[]>();
        layers = new Layer[numLayers + (sharedLayers?.Length ?? 0)];

        // Copy shared layers if provided
        int startIndex = 0;
        if (sharedLayers != null)
        {
            for (int i = 0; i < sharedLayers.Length; i++)
            {
                layers[i] = sharedLayers[i];
                float[] emptyOutputs = new float[sharedLayers[i].weights.GetLength(0)];
                currentBiasDerivatives.Add(emptyOutputs);
            }
            startIndex = sharedLayers.Length;
        }

        // Create remaining layers (except the last one)
        for (int i = startIndex; i < layers.Length - 1; i++)
        {
            int inputSize;
            if (i == startIndex) {
                // First layer after shared layers (or first layer if no shared layers)
                inputSize = sharedLayers != null ? 
                    sharedLayers[sharedLayers.Length - 1].weights.GetLength(0) : 
                    shape[0];
            } else {
                // Subsequent layers
                inputSize = shape[i - startIndex - 1];
            }
            
            int outputSize = shape[i - startIndex];
            Debug.Log($"Creating layer {i} with input size {inputSize} and output size {outputSize}");
            layers[i] = new Layer(inputSize, outputSize, activationFunction);
            float[] emptyOutputs = new float[outputSize];
            currentBiasDerivatives.Add(emptyOutputs);
        }

        // Create output layer
        int lastLayerIndex = layers.Length - 1;
        int finalInputSize = (lastLayerIndex == startIndex) ? 
            (sharedLayers != null ? sharedLayers[sharedLayers.Length - 1].weights.GetLength(0) : shape[0]) : 
            shape[shape.Length - 2];
        int finalOutputSize = shape[shape.Length - 1];
        
        Debug.Log($"Creating final layer with input size {finalInputSize} and output size {finalOutputSize}");
        layers[lastLayerIndex] = new Layer(finalInputSize, finalOutputSize, outputActivation);
        float[] empty = new float[finalOutputSize];
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

    public void SaveNetwork(string filePath, int startLayer = 0, int sharedLayerCount = 0)
    {
        var networkData = new List<LayerData>();

        // Save all layers, marking which ones are shared
        for (int i = 0; i < layers.Length; i++)
        {
            // Only save shared layers in the first network that contains them
            if (i >= startLayer && (sharedLayerCount == 0 || i >= sharedLayerCount))
            {
                var layer = layers[i];
                var layerData = new LayerData
                {
                    WeightsRows = layer.weights.GetLength(0),
                    WeightsCols = layer.weights.GetLength(1),
                    Weights = new float[layer.weights.Length],
                    Biases = (float[])layer.biases.Clone()
                };

                // Flatten the 2D weights array
                int index = 0;
                for (int j = 0; j < layer.weights.GetLength(0); j++)
                {
                    for (int k = 0; k < layer.weights.GetLength(1); k++)
                    {
                        layerData.Weights[index++] = layer.weights[j, k];
                    }
                }

                networkData.Add(layerData);
            }
        }

        string json = JsonUtility.ToJson(new NetworkData { Layers = networkData.ToArray() }, true);
        File.WriteAllText("/Users/leoschoolwork/Desktop/Personal/Projects/Coding/Unity/Connect Four/Assets/Data/" + filePath, json);
    }

    public void LoadNetwork(string filePath, int startLayer = 0, int sharedLayerCount = 0)
    {
        string json = File.ReadAllText("/Users/leoschoolwork/Desktop/Personal/Projects/Coding/Unity/Connect Four/Assets/Data/" + filePath);
        NetworkData networkData = JsonUtility.FromJson<NetworkData>(json);

        int dataIndex = 0;
        for (int i = startLayer; i < layers.Length; i++)
        {
            // Skip shared layers when loading the second network
            if (sharedLayerCount > 0 && i < sharedLayerCount && startLayer > 0)
                continue;

            if (dataIndex >= networkData.Layers.Length)
                break;

            var layerData = networkData.Layers[dataIndex++];

            // Reconstruct the 2D weights array
            layers[i].weights = new float[layerData.WeightsRows, layerData.WeightsCols];
            int index = 0;
            for (int j = 0; j < layerData.WeightsRows; j++)
            {
                for (int k = 0; k < layerData.WeightsCols; k++)
                {
                    layers[i].weights[j, k] = layerData.Weights[index++];
                }
            }

            // Load biases
            for (int j = 0; j < layerData.Biases.Length; j++)
            {
                layers[i].biases[j] = layerData.Biases[j];
            }
        }
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
            float gradient = 2 / output.Length * (output[i] - expectedOutput[i]);
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

    public void ApplyUniqueLayerGradients(float learningRate, int startFromLayer)
    {
        for (int i = startFromLayer; i < layers.Length; i++)
        {
            layers[i].ApplyGradients(learningRate);
        }
    }
}
