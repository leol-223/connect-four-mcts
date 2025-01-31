using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;  // For BinaryFormatter
using System.Linq;  // For Max(), Min() in forward pass, if needed
using System.Threading;  // For ThreadLocal

// Enums for activation and error types
public enum ActivationType
{
    ReLU,
    Softmax,
    Tanh
}

public enum ErrorType
{
    MeanSquaredError,
    CategoricalCrossEntropy
}

[Serializable]
public class NeuralNetwork
{
    private readonly int[] layers;        // Made readonly as it never changes after construction
    private readonly object lockObject = new object();  // Add lock object for thread safety
    
    private float[][] biases;    // biases[layer][neuron]
    private float[][][] weights; // weights[layer][neuron][previousLayerNeuron]

    private ActivationType hiddenActivation;
    private ActivationType outputActivation;
    private ErrorType errorType;

    [NonSerialized]
    private ThreadLocal<Random> rnd;  // Make Random thread-local

    public NeuralNetwork(int[] layers,
                         ActivationType hiddenActivation,
                         ActivationType outputActivation,
                         ErrorType errorType)
    {
        this.layers = layers;
        this.hiddenActivation = hiddenActivation;
        this.outputActivation = outputActivation;
        this.errorType = errorType;

        // Initialize thread-local random
        rnd = new ThreadLocal<Random>(() => new Random(Guid.NewGuid().GetHashCode()));

        // Initialize weights and biases
        biases = new float[layers.Length][];
        weights = new float[layers.Length][][];

        // We don't use index 0 for biases & weights because input layer has no incoming weights
        for (int i = 1; i < layers.Length; i++)
        {
            biases[i] = new float[layers[i]];
            weights[i] = new float[layers[i]][];

            // Initialize biases randomly
            for (int neuron = 0; neuron < layers[i]; neuron++)
            {
                biases[i][neuron] = GetRandomValue();
            }

            // Initialize weights randomly
            for (int neuron = 0; neuron < layers[i]; neuron++)
            {
                weights[i][neuron] = new float[layers[i - 1]];
                for (int prevNeuron = 0; prevNeuron < layers[i - 1]; prevNeuron++)
                {
                    weights[i][neuron][prevNeuron] = GetRandomValue();
                }
            }
        }
    }

    /// <summary>
    /// Forward pass through the network - Thread-safe version
    /// </summary>
    public float[] Forward(float[] input)
    {
        // Take a snapshot of weights and biases to ensure consistency during the forward pass
        float[][] biasesSnapshot;
        float[][][] weightsSnapshot;
        
        lock (lockObject)
        {
            biasesSnapshot = biases.Select(layer => layer?.ToArray()).ToArray();
            weightsSnapshot = weights.Select(layer => 
                layer?.Select(neuron => neuron?.ToArray()).ToArray()
            ).ToArray();
        }

        float[] activations = (float[])input.Clone();

        // Use snapshots instead of direct field access
        for (int layer = 1; layer < layers.Length; layer++)
        {
            float[] newActivations = new float[layers[layer]];
            ActivationType actType = (layer < layers.Length - 1)
                ? hiddenActivation
                : outputActivation;

            for (int neuron = 0; neuron < layers[layer]; neuron++)
            {
                float sum = biasesSnapshot[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < layers[layer - 1]; prevNeuron++)
                {
                    sum += activations[prevNeuron] * weightsSnapshot[layer][neuron][prevNeuron];
                }
                newActivations[neuron] = ApplyActivation(sum, actType);
            }

            if (actType == ActivationType.Softmax)
            {
                newActivations = Softmax(newActivations);
            }

            activations = newActivations;
        }

        return activations;
    }

    /// <summary>
    /// Train for a single epoch using batch-based updates
    /// </summary>
    /// <param name="inputs">Array of input vectors</param>
    /// <param name="targets">Array of target vectors</param>
    /// <param name="batchSize">Batch size</param>
    /// <param name="learningRate">Learning rate</param>
    public void Train(float[][] inputs, float[][] targets, int batchSize, float learningRate)
    {
        // Ensure thread-local random is initialized
        if (rnd.Value == null)
        {
            rnd = new ThreadLocal<Random>(() => new Random(Guid.NewGuid().GetHashCode()));
        }

        lock (lockObject)  // Lock during training to prevent concurrent modifications
        {
            // Shuffle data
            ShuffleData(inputs, targets);

            // Process each batch
            for (int batchStart = 0; batchStart < inputs.Length; batchStart += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, inputs.Length - batchStart);

                // Accumulate gradients
                var nablaB = CreateBiasesStorage();
                var nablaW = CreateWeightsStorage();

                // Compute grad for each sample in batch
                for (int i = 0; i < actualBatchSize; i++)
                {
                    int idx = batchStart + i;
                    var (deltaNablaB, deltaNablaW) = Backprop(inputs[idx], targets[idx]);

                    // Accumulate for the batch
                    for (int l = 1; l < layers.Length; l++)
                    {
                        for (int n = 0; n < layers[l]; n++)
                        {
                            nablaB[l][n] += deltaNablaB[l][n];
                            for (int pn = 0; pn < layers[l - 1]; pn++)
                            {
                                nablaW[l][n][pn] += deltaNablaW[l][n][pn];
                            }
                        }
                    }
                }

                // Update weights and biases with average gradient for the batch
                for (int l = 1; l < layers.Length; l++)
                {
                    for (int n = 0; n < layers[l]; n++)
                    {
                        biases[l][n] -= (learningRate / actualBatchSize) * nablaB[l][n];
                        for (int pn = 0; pn < layers[l - 1]; pn++)
                        {
                            weights[l][n][pn] -= (learningRate / actualBatchSize) * nablaW[l][n][pn];
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Backpropagation for one sample
    /// </summary>
    private (float[][], float[][][]) Backprop(float[] input, float[] target)
    {
        // Step 1: Forward pass (store activations & z-values for each layer)
        float[][] activations = new float[layers.Length][];
        float[][] zValues = new float[layers.Length][];

        activations[0] = (float[])input.Clone();

        for (int l = 1; l < layers.Length; l++)
        {
            activations[l] = new float[layers[l]];
            zValues[l] = new float[layers[l]];

            ActivationType actType = (l < layers.Length - 1) ? hiddenActivation : outputActivation;

            for (int n = 0; n < layers[l]; n++)
            {
                float sum = biases[l][n];
                for (int pn = 0; pn < layers[l - 1]; pn++)
                {
                    sum += activations[l - 1][pn] * weights[l][n][pn];
                }
                zValues[l][n] = sum; // Weighted sum before activation
            }

            // Softmax for last layer
            if (actType == ActivationType.Softmax && l == layers.Length - 1)
            {
                activations[l] = Softmax(zValues[l]);
            }
            else
            {
                for (int n = 0; n < layers[l]; n++)
                {
                    activations[l][n] = ApplyActivation(zValues[l][n], actType);
                }
            }
        }

        // Step 2: Output error
        float[] delta = new float[layers[layers.Length - 1]];
        float[] output = activations[layers.Length - 1];

        for (int n = 0; n < delta.Length; n++)
        {
            float errorDerivative = ErrorDerivative(output[n], target[n], errorType);
            
            float activationDerivative;
            if (outputActivation == ActivationType.Softmax &&
                errorType == ErrorType.CategoricalCrossEntropy)
            {
                // For Softmax + CrossEntropy, derivative simplifies to (output - target)
                activationDerivative = 1.0f;
            }
            else
            {
                activationDerivative = ActivationDerivative(zValues[layers.Length - 1][n], outputActivation);
            }
            delta[n] = errorDerivative * activationDerivative;
        }

        // Grad storage
        float[][] nablaB = CreateBiasesStorage();
        float[][][] nablaW = CreateWeightsStorage();

        // Step 2b: assign last-layer gradients
        for (int n = 0; n < layers[layers.Length - 1]; n++)
        {
            nablaB[layers.Length - 1][n] = delta[n];
            for (int pn = 0; pn < layers[layers.Length - 2]; pn++)
            {
                nablaW[layers.Length - 1][n][pn] = delta[n] * activations[layers.Length - 2][pn];
            }
        }

        // Step 3: Propagate backward
        for (int l = layers.Length - 2; l >= 1; l--)
        {
            float[] newDelta = new float[layers[l]];
            for (int n = 0; n < layers[l]; n++)
            {
                float error = 0f;
                for (int nn = 0; nn < layers[l + 1]; nn++)
                {
                    error += weights[l + 1][nn][n] * delta[nn];
                }
                float derivative = ActivationDerivative(zValues[l][n], hiddenActivation);
                newDelta[n] = error * derivative;
            }
            delta = newDelta;

            for (int n = 0; n < layers[l]; n++)
            {
                nablaB[l][n] = delta[n];
                for (int pn = 0; pn < layers[l - 1]; pn++)
                {
                    nablaW[l][n][pn] = delta[n] * activations[l - 1][pn];
                }
            }
        }

        return (nablaB, nablaW);
    }

    /// <summary>
    /// Save the entire network object to file using BinaryFormatter.
    /// </summary>
    public void Save(string filePath)
    {
        // Because of potential security vulnerabilities in BinaryFormatter,
        // do not use in untrusted scenarios. This is for demonstration only.
        BinaryFormatter bf = new BinaryFormatter();
        using (FileStream fs = new FileStream(filePath, FileMode.Create))
        {
            bf.Serialize(fs, this);
        }
    }

    /// <summary>
    /// Load the entire network object from file using BinaryFormatter.
    /// </summary>
    public static NeuralNetwork Load(string filePath)
    {
        BinaryFormatter bf = new BinaryFormatter();
        using (FileStream fs = new FileStream(filePath, FileMode.Open))
        {
            NeuralNetwork nn = (NeuralNetwork)bf.Deserialize(fs);
            nn.rnd = new ThreadLocal<Random>(() => new Random(Guid.NewGuid().GetHashCode()));
            return nn;
        }
    }

    // ----- Helper Methods -----

    private float GetRandomValue()
    {
        return (float)(rnd.Value.NextDouble() - 0.5);
    }

    private float[][] CreateBiasesStorage()
    {
        float[][] storage = new float[layers.Length][];
        for (int i = 0; i < layers.Length; i++)
        {
            storage[i] = new float[layers[i]];
        }
        return storage;
    }

    private float[][][] CreateWeightsStorage()
    {
        float[][][] storage = new float[layers.Length][][];
        for (int i = 0; i < layers.Length; i++)
        {
            storage[i] = new float[layers[i]][];
            for (int j = 0; j < layers[i]; j++)
            {
                // The 0th layer has no previous
                int prevCount = (i == 0) ? 0 : layers[i - 1];
                storage[i][j] = new float[prevCount];
            }
        }
        return storage;
    }

    // Shuffle inputs and targets together using Fisher-Yates
    private void ShuffleData(float[][] inputs, float[][] targets)
    {
        for (int i = inputs.Length - 1; i > 0; i--)
        {
            int swapIndex = rnd.Value.Next(i + 1);
            var tempIn = inputs[i];
            var tempOut = targets[i];
            inputs[i] = inputs[swapIndex];
            targets[i] = targets[swapIndex];
            inputs[swapIndex] = tempIn;
            targets[swapIndex] = tempOut;
        }
    }

    // Apply activation
    private float ApplyActivation(float x, ActivationType actType)
    {
        switch (actType)
        {
            case ActivationType.ReLU:
                return x > 0f ? x : 0f;
            case ActivationType.Softmax:
                // Typically handled as a vector, but fallback is exp(x)
                return (float)Math.Exp(x);
            case ActivationType.Tanh:
                return (float)Math.Tanh(x);
            default:
                return x;
        }
    }

    // Activation Derivative
    private float ActivationDerivative(float x, ActivationType actType)
    {
        switch (actType)
        {
            case ActivationType.ReLU:
                return (x > 0f) ? 1f : 0f;
            case ActivationType.Softmax:
                // Typically used with cross-entropy; derivative is simplified
                // We'll return 1.0f as a fallback
                return 1f;
            case ActivationType.Tanh:
                // derivative of tanh(x) is 1 - tanh²(x)
                float tanhX = (float)Math.Tanh(x);
                return 1f - (tanhX * tanhX);
            default:
                return 1f;
        }
    }

    // Softmax across a vector
    private float[] Softmax(float[] z)
    {
        // for numerical stability, subtract max
        float maxVal = z.Max();
        float sumExp = 0f;
        float[] exps = new float[z.Length];
        for (int i = 0; i < z.Length; i++)
        {
            exps[i] = (float)Math.Exp(z[i] - maxVal);
            sumExp += exps[i];
        }
        for (int i = 0; i < z.Length; i++)
        {
            exps[i] /= sumExp;
        }
        return exps;
    }

    // Error derivative wrt output
    private float ErrorDerivative(float output, float target, ErrorType errorType)
    {
        switch (errorType)
        {
            case ErrorType.MeanSquaredError:
                // d/dx (1/2 * (output - target)^2) => (output - target)
                return (output - target);
            case ErrorType.CategoricalCrossEntropy:
                // Typically, derivative ~ (output - target) if output is from softmax
                return (output - target);
            default:
                return (output - target);
        }
    }
}
