using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System;
using System.Linq;
using Unity.Mathematics;
using Unity.Burst;
using Unity.Collections;
using System.Threading;
using System.Diagnostics;
using Unity.Jobs;


public delegate float[] ActivationFunction(float[] inputs);
public delegate float[] ActivationDerivative(float[] inputs, float[] prevGradients);
public delegate float ErrorFunction(float[] outputs, float[] expectedOutputs);
public delegate float[] ErrorDerivative(float[] outputs, float[] expectedOutputs);

[Serializable]
public abstract class LayerData
{
    [SerializeField]
    public float[] Biases;
}

[Serializable]
public class DenseLayerData : LayerData
{
    [SerializeField]
    public float[,] Weights; // Changed from float[] to float[,]
    [SerializeField]
    public int WeightsRows;
    [SerializeField]
    public int WeightsCols;
}

[Serializable]
public class ConvolutionalLayerData : LayerData
{
    [SerializeField]
    public float[][][][] Filters; // [numFilters][depth][height][width]
    [SerializeField]
    public int FilterDepth;
    [SerializeField]
    public int FilterHeight;
    [SerializeField]
    public int FilterWidth;
    [SerializeField]
    public int Stride;
    [SerializeField]
    public bool UsePadding;
}

[Serializable]
public class PoolingLayerData : LayerData
{
    [SerializeField]
    public int PoolSize;
    [SerializeField]
    public int Stride;
    [SerializeField]
    public bool UseMaxPooling; // true for max pooling, false for average pooling
}

[Serializable]
public class NetworkData
{
    [SerializeField]
    public List<LayerData> Layers = new List<LayerData>();
    [SerializeField]
    public List<string> LayerTypes = new List<string>();
}

public enum LayerType
{
    Dense,
    Convolutional,
    Pooling
}

public class LayerConfig
{
    // Common properties
    public LayerType Type { get; set; }
    public int OutputSize { get; set; }
    public ActivationFunction Activation { get; set; }

    // Dense-specific properties
    public int InputSize { get; set; }

    // Convolutional-specific properties
    public int InputDepth { get; set; }
    public int InputHeight { get; set; }
    public int InputWidth { get; set; }
    public int NumFilters { get; set; }
    public int FilterSize { get; set; }
    public int Stride { get; set; }
    public bool UsePadding { get; set; }

    // Pooling-specific properties
    public int PoolSize { get; set; }
    public bool UseMaxPooling { get; set; }
}

// Add this new class to pair activation functions with their derivatives
public class Activation
{
    public ActivationFunction Function { get; private set; }
    public ActivationDerivative Derivative { get; private set; }

    public Activation(ActivationFunction function, ActivationDerivative derivative)
    {
        Function = function;
        Derivative = derivative;
    }

    // Static instances for common activation functions
    public static readonly Activation ReLU = new Activation(
        NeuralNetwork.ReLU,
        NeuralNetwork.ReLUDerivative
    );

    public static readonly Activation Sigmoid = new Activation(
        NeuralNetwork.Sigmoid,
        NeuralNetwork.SigmoidDerivative
    );

    public static readonly Activation Tanh = new Activation(
        NeuralNetwork.Tanh,
        NeuralNetwork.TanhDerivative
    );

    public static readonly Activation Softmax = new Activation(
        NeuralNetwork.Softmax,
        NeuralNetwork.SoftmaxDerivative
    );

    public static readonly Activation Linear = new Activation(
        NeuralNetwork.Linear,
        NeuralNetwork.LinearDerivative
    );

    public static readonly Activation LeakyReLU = new Activation(
        NeuralNetwork.LeakyReLU,
        NeuralNetwork.LeakyReLUDerivative
    );
}

public abstract class Layer : IDisposable
{
    protected int numInputs;
    protected int numOutputs;
    protected Activation activation;
    public float[] preActivationOutput;
    public float[] postActivationOutput;
    public float[] biases;
    public float[] biasGradients;
    
    protected Layer(int numInputs, int numOutputs, Activation activation) {
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.activation = activation;
        InitializeArrays();
    }

    protected virtual void InitializeArrays() {
        biases = new float[numOutputs];
        biasGradients = new float[numOutputs];
        preActivationOutput = new float[numOutputs];
        postActivationOutput = new float[numOutputs];
    }

    public abstract float[] Forward(float[] inputs);
    public abstract float[] Backward(float[] gradients, float[] prevLayerOutputs, ActivationDerivative activationDerivative);
    public abstract void ApplyGradients(float learningRate);
    
    public virtual void ResetGradients() {
        for (int i = 0; i < numOutputs; i++) {
            biasGradients[i] = 0;
        }
    }

    public abstract LayerData GetLayerData();
    public abstract void LoadLayerData(LayerData data);

    public abstract int GetOutputSize();
    public ActivationFunction GetActivationFunction() => activation.Function;
    public ActivationDerivative GetActivationDerivative() => activation.Derivative;

    protected abstract void Dispose(bool disposing);
    
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }
}

public class Dense : Layer {
    private float[,] weights;
    private float[,] weightGradients;
    private static System.Random random = new System.Random();
    
    private static float RandomNormal(float mean = 0.0f, float stdDev = 1.0f) {
        float u1, u2;
        do {
            u1 = (float)random.NextDouble();
        } while (u1 == 0);
        u2 = (float)random.NextDouble();
        
        float randStandard = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Sin(2.0f * Mathf.PI * u2);
        return mean + stdDev * randStandard;
    }

    public Dense(int numInputs, int numOutputs, Activation activation) 
        : base(numInputs, numOutputs, activation) {
        weights = new float[numOutputs, numInputs];
        weightGradients = new float[numOutputs, numInputs];
        InitializeWeights();
    }

    private void InitializeWeights() {
        float stdDev;
        if (activation.Function == NeuralNetwork.ReLU) {
            stdDev = Mathf.Sqrt(2.0f / numInputs); // He initialization
        } else {
            stdDev = Mathf.Sqrt(2.0f / (numInputs + numOutputs)); // Xavier initialization
        }

        for (int i = 0; i < numOutputs; i++) {
            biases[i] = 0;
            for (int j = 0; j < numInputs; j++) {
                weights[i, j] = RandomNormal(0, stdDev);
            }
        }
    }

    public override float[] Forward(float[] inputs) {
        for (int i = 0; i < numOutputs; i++) {
            preActivationOutput[i] = biases[i];
            for (int j = 0; j < numInputs; j++) {
                preActivationOutput[i] += inputs[j] * weights[i, j];
            }
        }
        postActivationOutput = activation.Function(preActivationOutput);
        return postActivationOutput;
    }

    public override float[] Backward(float[] gradients, float[] prevLayerOutputs, ActivationDerivative activationDerivative) {
        float[] preActivationGradients = activationDerivative(preActivationOutput, gradients);
        float[] prevLayerGradients = new float[numInputs];

        // Calculate gradients for this layer
        for (int i = 0; i < numOutputs; i++) {
            biasGradients[i] += preActivationGradients[i];
            for (int j = 0; j < numInputs; j++) {
                weightGradients[i, j] += prevLayerOutputs[j] * preActivationGradients[i];
            }
        }

        // Calculate gradients for previous layer
        for (int i = 0; i < numInputs; i++) {
            float gradient = 0;
            for (int j = 0; j < numOutputs; j++) {
                gradient += weights[j, i] * preActivationGradients[j];
            }
            prevLayerGradients[i] = gradient;
        }

        return prevLayerGradients;
    }

    public override void ApplyGradients(float learningRate) {
        for (int i = 0; i < numOutputs; i++) {
            biases[i] -= biasGradients[i] * learningRate;
            biasGradients[i] = 0;
            for (int j = 0; j < numInputs; j++) {
                weights[i, j] -= weightGradients[i, j] * learningRate;
                weightGradients[i, j] = 0;
            }
        }
    }

    public override void ResetGradients() {
        base.ResetGradients();
        for (int i = 0; i < numOutputs; i++) {
            for (int j = 0; j < numInputs; j++) {
                weightGradients[i, j] = 0;
            }
        }
    }

    public override LayerData GetLayerData()
    {
        DenseLayerData data = new DenseLayerData();
        data.WeightsRows = weights.GetLength(0);
        data.WeightsCols = weights.GetLength(1);
        data.Weights = weights;
        data.Biases = biases;

        return data;
    }

    public override void LoadLayerData(LayerData data)
    {
        if (!(data is DenseLayerData denseData))
        {
            throw new ArgumentException("Invalid layer data type");
        }

        // Initialize weights with the loaded dimensions
        weights = denseData.Weights; // Direct assignment of the 2D array
        biases = new float[denseData.Biases.Length];
        Array.Copy(denseData.Biases, biases, denseData.Biases.Length);
    }

    public override int GetOutputSize() => numOutputs;

    protected override void Dispose(bool disposing)
    {
        // No resources to dispose
    }
}

public class ConvolutionalLayer : Layer
{
    private NativeArray<float> filters;
    private NativeArray<float> filterGradients;
    private ThreadLocal<NativeArray<float>> threadLocalInput3D;
    private readonly int numFilters;
    private int inputDepth, inputHeight, inputWidth;
    private int outputDepth, outputHeight, outputWidth;
    private int filterSize; // Assuming cubic filter (same size in all dimensions)
    private int stride;
    private bool usePadding;
    private int padding;
    private readonly object lockObject = new object();

    
    public ConvolutionalLayer(
        int inputDepth, int inputHeight, int inputWidth,
        int numFilters, int filterSize, int stride, bool usePadding,
        Activation activation) 
        : base(inputDepth * inputHeight * inputWidth, 
               CalculateOutputSize(inputDepth, inputHeight, inputWidth, numFilters, filterSize, stride, usePadding), 
               activation)
    {
        this.inputDepth = inputDepth;
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.filterSize = filterSize;
        this.stride = stride;
        this.usePadding = usePadding;
        this.padding = usePadding ? filterSize / 2 : 0;

        // Calculate output dimensions
        outputDepth = numFilters;
        outputHeight = usePadding ? inputHeight : (inputHeight - filterSize) / stride + 1;
        outputWidth = usePadding ? inputWidth : (inputWidth - filterSize) / stride + 1;

        // Allocate persistent buffers
        int totalFilterSize = numFilters * inputDepth * this.filterSize * this.filterSize;
        filters = new NativeArray<float>(totalFilterSize, Allocator.Persistent);
        filterGradients = new NativeArray<float>(totalFilterSize, Allocator.Persistent);
        
        // Create thread-local input buffer
        threadLocalInput3D = new ThreadLocal<NativeArray<float>>(() => 
            new NativeArray<float>(inputDepth * inputHeight * inputWidth, Allocator.Persistent), true);
        
        InitializeFilters();
    }

    private static int CalculateOutputSize(int inputDepth, int inputHeight, int inputWidth, 
                                         int numFilters, int filterSize, int stride, bool usePadding)
    {
        int outHeight = usePadding ? inputHeight : (inputHeight - filterSize) / stride + 1;
        int outWidth = usePadding ? inputWidth : (inputWidth - filterSize) / stride + 1;
        return numFilters * outHeight * outWidth;
    }

    private void InitializeFilters()
    {
        // He initialization
        float stdDev = Mathf.Sqrt(2.0f / (filterSize * filterSize * filterSize * inputDepth));
        
        for (int f = 0; f < numFilters; f++)
        {
            for (int d = 0; d < inputDepth; d++)
            {
                for (int i = 0; i < filterSize; i++)
                {
                    for (int j = 0; j < filterSize; j++)
                    {
                        filters[f * inputDepth * filterSize * filterSize + d * filterSize * filterSize + i * filterSize + j] = RandomNormal(0, stdDev);
                    }
                }
            }
        }
    }

    private static float RandomNormal(float mean = 0.0f, float stdDev = 1.0f)
    {
        System.Random random = new System.Random();
        float u1, u2;
        do
        {
            u1 = (float)random.NextDouble();
        } while (u1 == 0);
        u2 = (float)random.NextDouble();
        
        float randStandard = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Sin(2.0f * Mathf.PI * u2);
        return mean + stdDev * randStandard;
    }

    [BurstCompile]
    private struct ConvolutionJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float> input;
        [ReadOnly] public NativeArray<float> filters;
        [ReadOnly] public NativeArray<float> biases;
        public NativeArray<float> output;
        
        public int inputDepth, inputHeight, inputWidth;
        public int filterSize, stride, padding;
        public int outputHeight, outputWidth;
        
        public void Execute(int index)
        {
            int f = index / (outputHeight * outputWidth);
            int outY = (index / outputWidth) % outputHeight;
            int outX = index % outputWidth;
            
            float sum = biases[f];
            int inY = outY * stride - padding;
            int inX = outX * stride - padding;
            
            // Unroll inner loops for better performance
            int filterOffset = f * inputDepth * filterSize * filterSize;
            for (int d = 0; d < inputDepth; d++)
            {
                int inputOffset = d * inputHeight * inputWidth;
                int filterDepthOffset = filterOffset + d * filterSize * filterSize;
                
                for (int i = 0; i < filterSize; i++)
                {
                    int y = inY + i;
                    if (y >= 0 && y < inputHeight)
                    {
                        int inputRowOffset = inputOffset + y * inputWidth;
                        int filterRowOffset = filterDepthOffset + i * filterSize;
                        
                        for (int j = 0; j < filterSize; j++)
                        {
                            int x = inX + j;
                            if (x >= 0 && x < inputWidth)
                            {
                                sum += input[inputRowOffset + x] * 
                                      filters[filterRowOffset + j];
                            }
                        }
                    }
                }
            }
            
            output[index] = sum;
        }
    }

    public override float[] Forward(float[] inputs)
    {
        // Get thread-local input buffer
        var input3D = threadLocalInput3D.Value;
        input3D.CopyFrom(inputs);
        
        var job = new ConvolutionJob
        {
            input = input3D,
            filters = filters,
            biases = new NativeArray<float>(biases, Allocator.TempJob),
            output = new NativeArray<float>(preActivationOutput.Length, Allocator.TempJob),
            inputDepth = inputDepth,
            inputHeight = inputHeight,
            inputWidth = inputWidth,
            filterSize = filterSize,
            stride = stride,
            padding = padding,
            outputHeight = outputHeight,
            outputWidth = outputWidth
        };

        // Use lock to ensure thread safety during job execution
        lock (lockObject)
        {
            var handle = job.Schedule(preActivationOutput.Length, 64);
            handle.Complete(); // Ensure job is completed before proceeding

            job.output.CopyTo(preActivationOutput);
            
            job.biases.Dispose();
            job.output.Dispose();
        }
        
        postActivationOutput = activation.Function(preActivationOutput);
        return postActivationOutput;
    }

    public override float[] Backward(float[] gradients, float[] prevLayerOutputs, ActivationDerivative activationDerivative)
    {
        // Get gradients with respect to pre-activation outputs
        float[] preActivationGradients = activationDerivative(preActivationOutput, gradients);

        // Convert previous layer outputs (inputs to this layer) to 3D
        float[,,] input3D = new float[inputDepth, inputHeight, inputWidth];
        int idx = 0;
        for (int d = 0; d < inputDepth; d++)
        {
            for (int i = 0; i < inputHeight; i++)
            {
                for (int j = 0; j < inputWidth; j++)
                {
                    input3D[d, i, j] = prevLayerOutputs[idx++];
                }
            }
        }

        // Initialize gradients for previous layer
        float[,,] prevLayerGradients3D = new float[inputDepth, inputHeight, inputWidth];

        // For each filter
        for (int f = 0; f < filters.Length; f++)
        {
            // For each output position
            for (int outY = 0; outY < outputHeight; outY++)
            {
                for (int outX = 0; outX < outputWidth; outX++)
                {
                    int outputIndex = f * outputHeight * outputWidth + outY * outputWidth + outX;
                    float gradientAtOutput = preActivationGradients[outputIndex];

                    // Add to bias gradient
                    biasGradients[f] += gradientAtOutput;

                    // Calculate input positions (no padding offset if not using padding)
                    int inY = outY * stride;
                    int inX = outX * stride;

                    // Update filter gradients and compute input gradients
                    for (int d = 0; d < inputDepth; d++)
                    {
                        for (int i = 0; i < filterSize; i++)
                        {
                            for (int j = 0; j < filterSize; j++)
                            {
                                int y = inY + i;
                                int x = inX + j;

                                // Only process if within input bounds
                                if (y < inputHeight && x < inputWidth)
                                {
                                    // Gradient for filter
                                    filterGradients[f * inputDepth * filterSize * filterSize + d * filterSize * filterSize + i * filterSize + j] += input3D[d, y, x] * gradientAtOutput;
                                    
                                    // Gradient for input (rotate filter by 180 degrees)
                                    prevLayerGradients3D[d, y, x] += 
                                        filters[f * inputDepth * filterSize * filterSize + d * filterSize * filterSize + (filterSize - 1 - i) * filterSize + (filterSize - 1 - j)] * gradientAtOutput;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert 3D gradients back to 1D array
        float[] prevLayerGradients = new float[numInputs];
        idx = 0;
        for (int d = 0; d < inputDepth; d++)
        {
            for (int i = 0; i < inputHeight; i++)
            {
                for (int j = 0; j < inputWidth; j++)
                {
                    prevLayerGradients[idx++] = prevLayerGradients3D[d, i, j];
                }
            }
        }

        return prevLayerGradients;
    }

    public override void ApplyGradients(float learningRate)
    {
        // Apply gradients to filters
        for (int f = 0; f < filters.Length; f++)
        {
            // Apply bias gradients
            biases[f] -= biasGradients[f] * learningRate;
            biasGradients[f] = 0;

            // Apply filter gradients
            for (int d = 0; d < inputDepth; d++)
            {
                for (int i = 0; i < filterSize; i++)
                {
                    for (int j = 0; j < filterSize; j++)
                    {
                        filters[f * inputDepth * filterSize * filterSize + d * filterSize * filterSize + i * filterSize + j] -= filterGradients[f * inputDepth * filterSize * filterSize + d * filterSize * filterSize + i * filterSize + j] * learningRate;
                        filterGradients[f * inputDepth * filterSize * filterSize + d * filterSize * filterSize + i * filterSize + j] = 0;
                    }
                }
            }
        }
    }

    public override LayerData GetLayerData()
    {
        ConvolutionalLayerData data = new ConvolutionalLayerData();
        data.Filters = new float[numFilters][][][];
        for (int f = 0; f < numFilters; f++)
        {
            data.Filters[f] = new float[inputDepth][][];
            for (int d = 0; d < inputDepth; d++)
            {
                data.Filters[f][d] = new float[filterSize][];
                for (int i = 0; i < filterSize; i++)
                {
                    data.Filters[f][d][i] = new float[filterSize];
                }
            }
        }
        data.FilterDepth = inputDepth;
        data.FilterHeight = filterSize;
        data.FilterWidth = filterSize;
        data.Stride = stride;
        data.UsePadding = usePadding;
        data.Biases = new float[biases.Length];
        Array.Copy(biases, data.Biases, biases.Length);
        return data;
    }

    public override void LoadLayerData(LayerData data)
    {
        if (!(data is ConvolutionalLayerData convData))
        {
            throw new ArgumentException("Invalid layer data type");
        }

        for (int f = 0; f < numFilters; f++)
        {
            for (int d = 0; d < inputDepth; d++)
            {
                for (int i = 0; i < filterSize; i++)
                {
                    for (int j = 0; j < filterSize; j++)
                    {
                        convData.Filters[f][d][i][j] = filters[f * inputDepth * filterSize * filterSize + d * filterSize * filterSize + i * filterSize + j];
                    }
                }
            }
        }
        Array.Copy(convData.Biases, biases, convData.Biases.Length);
    }

    public override int GetOutputSize() => outputDepth * outputHeight * outputWidth;

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            filters.Dispose();
            filterGradients.Dispose();
            
            // Dispose thread-local buffers
            foreach (var buffer in threadLocalInput3D.Values)
            {
                if (buffer.IsCreated)
                    buffer.Dispose();
            }
            threadLocalInput3D.Dispose();
        }
    }
}

[BurstCompile]
struct PoolingJob : IJobParallelFor
{
    [ReadOnly] public NativeArray<float> input;
    public NativeArray<float> output;
    [NativeDisableParallelForRestriction]
    public NativeArray<int> maxIndices;
    
    public int inputDepth, inputHeight, inputWidth;
    public int outputHeight, outputWidth;
    public int poolSize, stride;
    public bool useMaxPooling;

    public void Execute(int index)
    {
        int d = index / (outputHeight * outputWidth);
        int outY = (index % (outputHeight * outputWidth)) / outputWidth;
        int outX = index % outputWidth;
        
        int startY = outY * stride;
        int startX = outX * stride;
        
        if (useMaxPooling)
        {
            float maxVal = float.MinValue;
            int maxIdx = 0;
            
            // Unrolled pooling loop for better performance
            for (int i = 0; i < poolSize * poolSize; i++)
            {
                int y = startY + (i / poolSize);
                int x = startX + (i % poolSize);
                
                if (y < inputHeight && x < inputWidth)
                {
                    float val = input[d * inputHeight * inputWidth + y * inputWidth + x];
                    if (val > maxVal)
                    {
                        maxVal = val;
                        maxIdx = i;
                    }
                }
            }
            
            output[index] = maxVal == float.MinValue ? 0 : maxVal;
            maxIndices[index] = maxIdx;
        }
        else
        {
            float sum = 0;
            int count = 0;
            
            // Unrolled averaging loop
            for (int i = 0; i < poolSize * poolSize; i++)
            {
                int y = startY + (i / poolSize);
                int x = startX + (i % poolSize);
                
                if (y < inputHeight && x < inputWidth)
                {
                    sum += input[d * inputHeight * inputWidth + y * inputWidth + x];
                    count++;
                }
            }
            
            output[index] = count > 0 ? sum / count : 0;
        }
    }
}

public class PoolingLayer : Layer
{
    private int inputDepth, inputHeight, inputWidth;
    private int outputHeight, outputWidth;
    private int poolSize;
    private int stride;
    private bool useMaxPooling;
    
    // Thread-local storage for intermediate calculations
    private ThreadLocal<float[,,]> threadLocalInput3D;
    private ThreadLocal<float[,,]> threadLocalOutput3D;
    private ThreadLocal<int[,,]> threadLocalMaxIndices;
    private readonly object lockObject = new object();
    
    public PoolingLayer(
        int inputDepth, int inputHeight, int inputWidth,
        int poolSize, int stride, bool useMaxPooling = true)
        : base(inputDepth * inputHeight * inputWidth,
               inputDepth * ((inputHeight - poolSize) / stride + 1) * ((inputWidth - poolSize) / stride + 1),
               Activation.Linear)
    {
        if (poolSize > inputHeight || poolSize > inputWidth)
        {
            throw new ArgumentException($"Pool size ({poolSize}) cannot be larger than input dimensions ({inputHeight}x{inputWidth})");
        }

        if (stride <= 0)
        {
            throw new ArgumentException($"Stride must be positive, got {stride}");
        }

        this.inputDepth = inputDepth;
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.poolSize = poolSize;
        this.stride = stride;
        this.useMaxPooling = useMaxPooling;

        // Calculate output dimensions
        outputHeight = (inputHeight - poolSize) / stride + 1;
        outputWidth = (inputWidth - poolSize) / stride + 1;

        if (outputHeight <= 0 || outputWidth <= 0)
        {
            throw new ArgumentException($"Invalid output dimensions: {outputHeight}x{outputWidth}. " +
                                      $"Check pool size ({poolSize}) and stride ({stride})");
        }

        // Initialize thread-local storage with proper factory methods
        threadLocalInput3D = new ThreadLocal<float[,,]>(() => 
            new float[inputDepth, inputHeight, inputWidth], true);
        threadLocalOutput3D = new ThreadLocal<float[,,]>(() => 
            new float[inputDepth, outputHeight, outputWidth], true);
        threadLocalMaxIndices = new ThreadLocal<int[,,]>(() => 
            new int[inputDepth, outputHeight, outputWidth], true);
    }

    public override float[] Forward(float[] inputs)
    {
        if (inputs.Length != inputDepth * inputHeight * inputWidth)
        {
            return new float[numOutputs];
        }

        var inputArray = new NativeArray<float>(inputs, Allocator.TempJob);
        var outputArray = new NativeArray<float>(numOutputs, Allocator.TempJob);
        var maxIndicesArray = useMaxPooling ? 
            new NativeArray<int>(numOutputs, Allocator.TempJob) : 
            new NativeArray<int>(0, Allocator.TempJob);

        var job = new PoolingJob
        {
            input = inputArray,
            output = outputArray,
            maxIndices = maxIndicesArray,
            inputDepth = inputDepth,
            inputHeight = inputHeight,
            inputWidth = inputWidth,
            outputHeight = outputHeight,
            outputWidth = outputWidth,
            poolSize = poolSize,
            stride = stride,
            useMaxPooling = useMaxPooling
        };

        // Schedule the job with automatic batch size
        var handle = job.Schedule(numOutputs, 64);
        handle.Complete();

        // Copy results
        var output = new float[numOutputs];
        outputArray.CopyTo(output);
        
        // Store max indices for backprop if needed
        if (useMaxPooling)
        {
            threadLocalMaxIndices.Value = maxIndicesArray;
        }

        // Cleanup
        inputArray.Dispose();
        outputArray.Dispose();
        if (useMaxPooling) maxIndicesArray.Dispose();

        postActivationOutput = output;
        return output;
    }

    public override float[] Backward(float[] gradients, float[] prevLayerOutputs, ActivationDerivative activationDerivative)
    {
        if (gradients.Length != numOutputs)
        {
            UnityEngine.Debug.LogError($"Invalid gradient dimensions. Expected {numOutputs}, got {gradients.Length}");
            return new float[numInputs];
        }

        lock (lockObject)
        {
            // Clear arrays before use
            float[,,] prevLayerGradients3D = new float[inputDepth, inputHeight, inputWidth];
            
            // Convert previous layer outputs to 3D
            float[,,] prevLayerOutputs3D = new float[inputDepth, inputHeight, inputWidth];
            int idx = 0;
            for (int d = 0; d < inputDepth; d++)
            {
                for (int i = 0; i < inputHeight; i++)
                {
                    for (int j = 0; j < inputWidth; j++)
                    {
                        prevLayerOutputs3D[d, i, j] = prevLayerOutputs[idx++];
                    }
                }
            }

            // Compute gradients
            for (int d = 0; d < inputDepth; d++)
            {
                for (int outY = 0; outY < outputHeight; outY++)
                {
                    for (int outX = 0; outX < outputWidth; outX++)
                    {
                        int startY = outY * stride;
                        int startX = outX * stride;

                        if (useMaxPooling)
                        {
                            // For max pooling, gradient flows only through the maximum value
                            int maxIdx = threadLocalMaxIndices.Value[d, outY, outX];
                            int maxY = startY + (maxIdx / poolSize);
                            int maxX = startX + (maxIdx % poolSize);

                            if (maxY < inputHeight && maxX < inputWidth)
                            {
                                prevLayerGradients3D[d, maxY, maxX] += gradients[d * outputHeight * outputWidth + outY * outputWidth + outX];
                            }
                        }
                        else
                        {
                            // For average pooling, gradient is distributed evenly
                            float gradient = gradients[d * outputHeight * outputWidth + outY * outputWidth + outX];
                            int count = 0;

                            // Count valid positions in the pooling window
                            for (int y = 0; y < poolSize; y++)
                            {
                                for (int x = 0; x < poolSize; x++)
                                {
                                    if (startY + y < inputHeight && startX + x < inputWidth)
                                    {
                                        count++;
                                    }
                                }
                            }

                            // Distribute gradient
                            if (count > 0)
                            {
                                float distributedGradient = gradient / count;
                                for (int y = 0; y < poolSize; y++)
                                {
                                    for (int x = 0; x < poolSize; x++)
                                    {
                                        if (startY + y < inputHeight && startX + x < inputWidth)
                                        {
                                            prevLayerGradients3D[d, startY + y, startX + x] += distributedGradient;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Convert gradients back to 1D array
            float[] prevLayerGradients = new float[numInputs];
            idx = 0;
            for (int d = 0; d < inputDepth; d++)
            {
                for (int i = 0; i < inputHeight; i++)
                {
                    for (int j = 0; j < inputWidth; j++)
                    {
                        prevLayerGradients[idx++] = prevLayerGradients3D[d, i, j];
                    }
                }
            }

            return prevLayerGradients;
        }
    }

    public override void ApplyGradients(float learningRate)
    {
        // Pooling layers have no parameters to update
    }

    public override LayerData GetLayerData()
    {
        return new PoolingLayerData
        {
            PoolSize = poolSize,
            Stride = stride,
            UseMaxPooling = useMaxPooling
        };
    }

    public override void LoadLayerData(LayerData data)
    {
        if (!(data is PoolingLayerData poolData))
        {
            throw new ArgumentException("Invalid layer data type");
        }

        poolSize = poolData.PoolSize;
        stride = poolData.Stride;
        useMaxPooling = poolData.UseMaxPooling;
    }

    public override int GetOutputSize() => inputDepth * outputHeight * outputWidth;

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            threadLocalInput3D?.Dispose();
            threadLocalOutput3D?.Dispose();
            threadLocalMaxIndices?.Dispose();
        }
    }
}

public class NeuralNetwork
{
    private List<Layer> layers;
    private ErrorFunction errorFunction;
    private ErrorDerivative errorDerivative;
    private List<float[]> currentBiasDerivatives;
    private const float EPSILON = 1e-7f;
    private const float ONE_MINUS_EPSILON = 1f - 1e-7f;

    public NeuralNetwork(ErrorFunction errorFunction, ErrorDerivative errorDerivative)
    {
        this.layers = new List<Layer>();
        this.currentBiasDerivatives = new List<float[]>();
        this.errorFunction = errorFunction;
        this.errorDerivative = errorDerivative;
    }

    public void AddDenseLayer(int inputSize, int outputSize, Activation activation = null)
    {
        // Default to ReLU if no activation is provided
        activation ??= Activation.ReLU;
        var layer = new Dense(inputSize, outputSize, activation);
        layers.Add(layer);
        currentBiasDerivatives.Add(new float[outputSize]);
    }

    public void AddConvolutionalLayer(
        int inputDepth, int inputHeight, int inputWidth,
        int numFilters, int filterSize, int stride = 1,
        bool usePadding = true, Activation activation = null)
    {
        // Default to ReLU if no activation is provided
        activation ??= Activation.ReLU;
        var layer = new ConvolutionalLayer(
            inputDepth, inputHeight, inputWidth,
            numFilters, filterSize, stride, usePadding, activation);
        layers.Add(layer);
        currentBiasDerivatives.Add(new float[layer.GetOutputSize()]);
    }

    public void AddPoolingLayer(
        int inputDepth, int inputHeight, int inputWidth,
        int poolSize, int stride = 2, bool useMaxPooling = true)
    {
        var layer = new PoolingLayer(
            inputDepth, inputHeight, inputWidth,
            poolSize, stride, useMaxPooling);
        layers.Add(layer);
        currentBiasDerivatives.Add(new float[layer.GetOutputSize()]);
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
        for (int i = 0; i < layers.Count; i++) {
            runningOutput = layers[i].Forward(runningOutput);
            // DebugArray("LAYER " + i.ToString() + " OUTPUT: ", runningOutput);
        }
        return runningOutput;
    }
    public float TrainOneEpoch(List<float[]> inputData, List<float[]> outputData, float learningRate, int batchSize, bool shuffle = true) {
        float totalCost = 0f;
        int batchCount = 0;
        
        // Create index array for shuffling
        int[] indices = Enumerable.Range(0, inputData.Count).ToArray();
        if (shuffle) {
            // Fisher-Yates shuffle
            for (int i = indices.Length - 1; i > 0; i--) {
                int j = UnityEngine.Random.Range(0, i + 1);
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }

        // Use shuffled indices to access data
        for (int i = 0; i < inputData.Count; i++) {
            int idx = indices[i];
            float[] input = inputData[idx];
            float[] output = outputData[idx];
            totalCost += CalculateGradients(input, output);
            batchCount += 1;
            if (batchCount >= batchSize) {
                ApplyGradients(learningRate, batchSize);
                batchCount = 0;
            }
        }
        
        // Handle remaining samples in last batch
        if (batchCount > 0) {
            ApplyGradients(learningRate, batchCount);  // Use actual batch count for last batch
        }
        
        return totalCost / inputData.Count;
    }

    public float CalculateCost(List<float[]> inputData, List<float[]> outputData) {
        float totalCost = 0f;
        for (int i = 0; i < inputData.Count; i++)
        {
            float[] input = inputData[i];
            float[] expectedOutput = outputData[i];
            float[] output = Evaluate(input);
            totalCost += errorFunction(output, expectedOutput);
        }
        return totalCost / inputData.Count;
    }

    public void ApplyGradients(float learningRate, int batchSize)
    {
        float lr = learningRate / batchSize;
        for (int i = 0; i < layers.Count; i++) {
            layers[i].ApplyGradients(lr);
        }
    }
    public void ResetGradients()
    {
        for (int i = 0; i < layers.Count; i++)
        {
            layers[i].ResetGradients();
        }
    }
    public float CalculateGradients(float[] input, float[] expectedOutput) {
        // Forward pass
        float[] output = Evaluate(input);
        float cost = errorFunction(output, expectedOutput);

        // Backward pass
        float[] gradients = errorDerivative(output, expectedOutput);
        
        // Propagate gradients through the network
        for (int i = layers.Count - 1; i >= 0; i--) {
            float[] prevLayerOutput = i > 0 ? layers[i - 1].postActivationOutput : input;
            gradients = layers[i].Backward(
                gradients, 
                prevLayerOutput, 
                layers[i].GetActivationDerivative()
            );
        }

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
            // Using min/max is slightly faster than Clamp for single bounds
            float clippedOutput = outputs[i] < EPSILON ? EPSILON : 
                                (outputs[i] > ONE_MINUS_EPSILON ? ONE_MINUS_EPSILON : outputs[i]);
            cost += (-expectedOutputs[i] * Mathf.Log(clippedOutput));
        }
        return cost / outputs.Length;
    }

    public static float[] CategoricalCrossEntropyDerivative(float[] output, float[] expectedOutput)
    {
        float[] outputGradients = new float[output.Length];
        for (int i = 0; i < output.Length; i++)
        {
            float clippedOutput = output[i] < EPSILON ? EPSILON : 
                                 (output[i] > ONE_MINUS_EPSILON ? ONE_MINUS_EPSILON : output[i]);
            outputGradients[i] = -expectedOutput[i] / clippedOutput;
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

    public static float[] Linear(float[] inputs)
    {
        return inputs;
    }

    public static float[] LinearDerivative(float[] inputs, float[] prevGradients)
    {
        return prevGradients;
    }

    public static float[] LeakyReLU(float[] inputs)
    {
        float[] outputs = new float[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            if (inputs[i] > 0) {
                outputs[i] = inputs[i];
            } else {
                outputs[i] = inputs[i] * 0.01f;
            }
        }
        return outputs;
    }

    public static float[] LeakyReLUDerivative(float[] inputs, float[] prevGradients)
    {
        float[] derivatives = new float[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            derivatives[i] = inputs[i] > 0 ? prevGradients[i] : -0.01f * prevGradients[i];
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

        // Find maximum input for numerical stability
        float maxInput = inputs[0];
        for (int i = 1; i < inputs.Length; i++) {
            maxInput = Mathf.Max(maxInput, inputs[i]);
        }

        for (int i = 0; i < inputs.Length; i++)
        {
            float power = Mathf.Exp(inputs[i] - maxInput);
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
        // Find maximum input for numerical stability
        float maxInput = inputs[0];
        for (int i = 1; i < inputs.Length; i++) {
            maxInput = Mathf.Max(maxInput, inputs[i]);
        }

        // Compute exponentials with numerical stability
        float[] exps = new float[inputs.Length];
        float s = 0;
        for (int i = 0; i < inputs.Length; i++) {
            float exp = Mathf.Exp(inputs[i] - maxInput);
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
        for (int i = startFromLayer; i < layers.Count; i++)
        {
            layers[i].ApplyGradients(learningRate);
        }
    }

    [Serializable]
    public class SerializableArray1D
    {
        public float[] data;
        public SerializableArray1D(float[] array) { data = array; }
    }

    [Serializable]
    public class SerializableArray2D
    {
        public float[] data;
        public int rows;
        public int cols;
        
        public SerializableArray2D(float[,] array)
        {
            rows = array.GetLength(0);
            cols = array.GetLength(1);
            data = new float[rows * cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    data[i * cols + j] = array[i, j];
        }
        
        public float[,] ToArray2D()
        {
            float[,] array = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    array[i, j] = data[i * cols + j];
            return array;
        }
    }

    [Serializable]
    public class SerializableArray4D
    {
        public float[] data;
        public int dim1, dim2, dim3, dim4;
        
        public SerializableArray4D(float[][][][] array)
        {
            dim1 = array.Length;
            dim2 = array[0].Length;
            dim3 = array[0][0].Length;
            dim4 = array[0][0][0].Length;
            
            data = new float[dim1 * dim2 * dim3 * dim4];
            int index = 0;
            for (int i = 0; i < dim1; i++)
                for (int j = 0; j < dim2; j++)
                    for (int k = 0; k < dim3; k++)
                        for (int l = 0; l < dim4; l++)
                            data[index++] = array[i][j][k][l];
        }
        
        public float[][][][] ToArray4D()
        {
            float[][][][] array = new float[dim1][][][];
            for (int i = 0; i < dim1; i++)
            {
                array[i] = new float[dim2][][];
                for (int j = 0; j < dim2; j++)
                {
                    array[i][j] = new float[dim3][];
                    for (int k = 0; k < dim3; k++)
                    {
                        array[i][j][k] = new float[dim4];
                    }
                }
            }
            
            int index = 0;
            for (int i = 0; i < dim1; i++)
                for (int j = 0; j < dim2; j++)
                    for (int k = 0; k < dim3; k++)
                        for (int l = 0; l < dim4; l++)
                            array[i][j][k][l] = data[index++];
            
            return array;
        }
    }

    [Serializable]
    public class SerializableDenseLayer
    {
        public SerializableArray2D weights;
        public SerializableArray1D biases;
        
        public SerializableDenseLayer(DenseLayerData layer)
        {
            weights = new SerializableArray2D(layer.Weights);
            biases = new SerializableArray1D(layer.Biases);
        }
    }

    [Serializable]
    public class SerializableConvLayer
    {
        public SerializableArray4D filters;
        public SerializableArray1D biases;
        public int filterDepth;
        public int filterHeight;
        public int filterWidth;
        public int stride;
        public bool usePadding;
        
        public SerializableConvLayer(ConvolutionalLayerData layer)
        {
            filters = new SerializableArray4D(layer.Filters);
            biases = new SerializableArray1D(layer.Biases);
            filterDepth = layer.FilterDepth;
            filterHeight = layer.FilterHeight;
            filterWidth = layer.FilterWidth;
            stride = layer.Stride;
            usePadding = layer.UsePadding;
        }
    }

    [Serializable]
    public class SerializablePoolingLayer
    {
        public int poolSize;
        public int stride;
        public bool useMaxPooling;
        
        public SerializablePoolingLayer(PoolingLayerData layer)
        {
            poolSize = layer.PoolSize;
            stride = layer.Stride;
            useMaxPooling = layer.UseMaxPooling;
        }
    }

    [Serializable]
    public class SerializableNetwork
    {
        public List<string> layerTypes = new List<string>();
        public List<string> serializedLayers = new List<string>();
    }

    public void SaveNetwork(string path)
    {
        SerializableNetwork data = new SerializableNetwork();
        
        for (int i = 0; i < layers.Count; i++)
        {
            LayerData layerData = layers[i].GetLayerData();
            data.layerTypes.Add(layers[i].GetType().Name);
            
            if (layerData is DenseLayerData denseLayer)
            {
                var serializableLayer = new SerializableDenseLayer(denseLayer);
                data.serializedLayers.Add(JsonUtility.ToJson(serializableLayer));
            }
            else if (layerData is ConvolutionalLayerData convLayer)
            {
                var serializableLayer = new SerializableConvLayer(convLayer);
                data.serializedLayers.Add(JsonUtility.ToJson(serializableLayer));
            }
            else if (layerData is PoolingLayerData poolLayer)
            {
                var serializableLayer = new SerializablePoolingLayer(poolLayer);
                data.serializedLayers.Add(JsonUtility.ToJson(serializableLayer));
            }
        }
        
        string json = JsonUtility.ToJson(data, true);
        File.WriteAllText(path, json);
    }

    public void LoadNetwork(string path)
    {
        if (!File.Exists(path))
        {
            UnityEngine.Debug.LogError($"No save file found at {path}");
            return;
        }

        string json = File.ReadAllText(path);
        SerializableNetwork data = JsonUtility.FromJson<SerializableNetwork>(json);
        
        if (data.layerTypes.Count != layers.Count)
        {
            UnityEngine.Debug.LogError("Saved network architecture doesn't match current network architecture");
            return;
        }

        for (int i = 0; i < layers.Count; i++)
        {
            if (data.layerTypes[i] != layers[i].GetType().Name)
            {
                UnityEngine.Debug.LogError($"Layer type mismatch at index {i}");
                return;
            }

            if (layers[i] is Dense)
            {
                var serializableLayer = JsonUtility.FromJson<SerializableDenseLayer>(data.serializedLayers[i]);
                var layerData = new DenseLayerData
                {
                    Weights = serializableLayer.weights.ToArray2D(),
                    Biases = serializableLayer.biases.data
                };
                layers[i].LoadLayerData(layerData);
            }
            else if (layers[i] is ConvolutionalLayer)
            {
                var serializableLayer = JsonUtility.FromJson<SerializableConvLayer>(data.serializedLayers[i]);
                var layerData = new ConvolutionalLayerData
                {
                    Filters = serializableLayer.filters.ToArray4D(),
                    Biases = serializableLayer.biases.data,
                    FilterDepth = serializableLayer.filterDepth,
                    FilterHeight = serializableLayer.filterHeight,
                    FilterWidth = serializableLayer.filterWidth,
                    Stride = serializableLayer.stride,
                    UsePadding = serializableLayer.usePadding
                };
                layers[i].LoadLayerData(layerData);
            }
            else if (layers[i] is PoolingLayer)
            {
                var serializableLayer = JsonUtility.FromJson<SerializablePoolingLayer>(data.serializedLayers[i]);
                var layerData = new PoolingLayerData
                {
                    PoolSize = serializableLayer.poolSize,
                    Stride = serializableLayer.stride,
                    UseMaxPooling = serializableLayer.useMaxPooling
                };
                layers[i].LoadLayerData(layerData);
            }
        }
    }
}