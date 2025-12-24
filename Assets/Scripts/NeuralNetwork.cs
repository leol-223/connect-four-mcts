using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using UnityEngine; // optional if using in Unity

[Serializable]
public class NeuralNetwork
{
    public enum ActivationType
    {
        Sigmoid,
        ReLU,
        Tanh,
        Softmax // Typically used for output layer in classification
    }

    public enum LossType
    {
        MSE,
        CrossEntropy
    }

    [Serializable]
    private class Layer
    {
        public float[,] Weights;  // [outputSize, inputSize]
        public float[] Biases;    // [outputSize]

        [NonSerialized] public float[] Inputs;   // Used in backprop
        [NonSerialized] public float[] Outputs;  // Used in backprop
    }

    private List<Layer> _layers;

    // Activation for hidden layers
    private ActivationType _hiddenActivation;
    // Activation for output layer
    private ActivationType _outputActivation;
    private LossType _loss;

    private System.Random _random;

    // --- Hyperparameters to mitigate numeric issues ---
    // Epsilon used in softmax to avoid 0 or 1
    public float SoftmaxEpsilon { get; set; } = 1e-7f;
    // Max norm for gradients in a batch
    public float GradientClipNorm { get; set; } = 5.0f;

    [NonSerialized]
    private object _lockObject = new object();

    /// <summary>
    /// Constructor
    /// </summary>
    /// <param name="layerSizes">Network layer sizes, e.g. [input, hidden1, hidden2, output]</param>
    /// <param name="hiddenActivation">Activation for hidden layers (Sigmoid/ReLU/Tanh/etc.)</param>
    /// <param name="outputActivation">Activation for final (output) layer</param>
    /// <param name="loss">Loss function</param>
    public NeuralNetwork(int[] layerSizes,
                         ActivationType hiddenActivation,
                         ActivationType outputActivation,
                         LossType loss)
    {
        if (layerSizes.Length < 2)
            throw new ArgumentException("Must have at least input and output layers.");

        _hiddenActivation = hiddenActivation;
        _outputActivation = outputActivation;
        _loss = loss;
        _random = new System.Random();

        _layers = new List<Layer>();
        for (int i = 1; i < layerSizes.Length; i++)
        {
            int inputSize = layerSizes[i - 1];
            int outputSize = layerSizes[i];

            var layer = new Layer
            {
                Weights = new float[outputSize, inputSize],
                Biases = new float[outputSize],
                Inputs = new float[inputSize],
                Outputs = new float[outputSize]
            };

            // Xavier/Glorot initialization
            float limit = (float)Math.Sqrt(6f / (inputSize + outputSize));
            for (int o = 0; o < outputSize; o++)
            {
                for (int x = 0; x < inputSize; x++)
                {
                    layer.Weights[o, x] = (float)(_random.NextDouble() * 2.0 - 1.0) * limit;
                }
                layer.Biases[o] = 0f;
            }

            _layers.Add(layer);
        }
    }

    #region Forward

    public float[] Forward(float[] input)
    {
        lock (_lockObject)
        {
            float[] current = input;
            for (int l = 0; l < _layers.Count; l++)
            {
                Layer layer = _layers[l];
                layer.Inputs = current;
                float[] rawOutput = new float[layer.Biases.Length];

                // Compute logits
                for (int o = 0; o < layer.Biases.Length; o++)
                {
                    float sum = layer.Biases[o];
                    for (int i = 0; i < current.Length; i++)
                    {
                        sum += layer.Weights[o, i] * current[i];
                    }
                    rawOutput[o] = sum;
                }

                // Decide which activation to use
                ActivationType act = (l == _layers.Count - 1) ? _outputActivation : _hiddenActivation;
                float[] activated = ApplyActivation(rawOutput, act);

                layer.Outputs = activated;
                current = activated;
            }
            return current;
        }
    }

    #endregion

    #region Train Single (Stochastic)

    /// <summary>
    /// Backprop for a single sample (immediate update).
    /// </summary>
    public void TrainSingleStochastic(float[] input, float[] target, float learningRate)
    {
        if (target.Length != _layers[_layers.Count - 1].Biases.Length)
            throw new ArgumentException("Target size mismatch with output layer.");

        lock (_lockObject)
        {
            float[] output = Forward(input);

            float[] outputError = ComputeOutputError(output, target, _loss, _outputActivation);

            // Backprop layers
            for (int l = _layers.Count - 1; l >= 0; l--)
            {
                Layer layer = _layers[l];
                ActivationType act = (l == _layers.Count - 1) ? _outputActivation : _hiddenActivation;

                float[] delta = new float[layer.Biases.Length];
                for (int o = 0; o < delta.Length; o++)
                {
                    if (l == _layers.Count-1 && _loss == LossType.CrossEntropy && _outputActivation == ActivationType.Softmax)
                    {
                        // softmax + crossentropy cancels out
                        delta[o] = outputError[o]; 
                    }
                    else
                    {
                        float dAct = DerivativeActivate(layer.Outputs[o], act);
                        delta[o] = outputError[o] * dAct;
                    }
                }

                // Update weights & biases
                for (int o = 0; o < delta.Length; o++)
                {
                    for (int i = 0; i < layer.Inputs.Length; i++)
                    {
                        float grad = delta[o] * layer.Inputs[i];
                        layer.Weights[o, i] -= learningRate * grad;
                    }
                    layer.Biases[o] -= learningRate * delta[o];
                }

                // Propagate error backward
                if (l > 0)
                {
                    float[] nextError = new float[_layers[l - 1].Biases.Length];
                    for (int i = 0; i < nextError.Length; i++)
                    {
                        float sum = 0f;
                        for (int o = 0; o < delta.Length; o++)
                        {
                            sum += layer.Weights[o, i] * delta[o];
                        }
                        nextError[i] = sum;
                    }
                    outputError = nextError;
                }
            }
        }
    }

    #endregion

    #region Mini-Batch / Epoch

    /// <summary>
    /// Train one epoch on the dataset, in mini-batches, with gradient clipping.
    /// </summary>
    public void TrainEpoch(float[][] inputs,
                           float[][] targets,
                           int batchSize,
                           float learningRate,
                           bool shuffle = true)
    {
        if (inputs.Length != targets.Length)
            throw new ArgumentException("Inputs and targets must have same length.");

        int numSamples = inputs.Length;
        if (shuffle) 
            ShuffleData(inputs, targets);

        var weightGrads = new float[_layers.Count][][];
        var biasGrads = new float[_layers.Count][];
        for (int l = 0; l < _layers.Count; l++)
        {
            int outSize = _layers[l].Biases.Length;
            int inSize = _layers[l].Inputs.Length;
            biasGrads[l] = new float[outSize];
            weightGrads[l] = new float[outSize][];
            for (int o = 0; o < outSize; o++) weightGrads[l][o] = new float[inSize];
        }

        for (int start = 0; start < numSamples; start += batchSize)
        {
            for(int l=0; l < _layers.Count; l++) {
                Array.Clear(biasGrads[l], 0, biasGrads[l].Length);
                for(int o=0; o < weightGrads[l].Length; o++) 
                    Array.Clear(weightGrads[l][o], 0, weightGrads[l][o].Length);
            }

            int end = Math.Min(start + batchSize, numSamples);
            int currentBatchSize = end - start;

            for (int l = 0; l < _layers.Count; l++)
            {
                int outSize = _layers[l].Biases.Length;
                int inSize  = _layers[l].Inputs.Length;

                weightGrads[l] = new float[outSize][];
                biasGrads[l]   = new float[outSize];

                for (int o = 0; o < outSize; o++)
                {
                    weightGrads[l][o] = new float[inSize];
                }
            }

            // Accumulate gradients for each sample in this batch
            for (int idx = start; idx < end; idx++)
            {
                float[] output = Forward(inputs[idx]);
                float[] outputError = ComputeOutputError(output, targets[idx], _loss, _outputActivation);

                // Backprop to accumulate
                for (int l = _layers.Count - 1; l >= 0; l--)
                {
                    Layer layer = _layers[l];
                    ActivationType act = (l == _layers.Count - 1) ? _outputActivation : _hiddenActivation;

                    float[] delta = new float[layer.Biases.Length];
                    for (int o = 0; o < delta.Length; o++)
                    {
                        float dAct = DerivativeActivate(layer.Outputs[o], act);
                        delta[o] = outputError[o] * dAct;
                    }

                    // accumulate grads
                    for (int o = 0; o < delta.Length; o++)
                    {
                        biasGrads[l][o] += delta[o];
                        for (int i = 0; i < layer.Inputs.Length; i++)
                        {
                            weightGrads[l][o][i] += delta[o] * layer.Inputs[i];
                        }
                    }

                    // Next error
                    if (l > 0)
                    {
                        float[] nextError = new float[_layers[l - 1].Biases.Length];
                        for (int i = 0; i < nextError.Length; i++)
                        {
                            float sum = 0f;
                            for (int o = 0; o < delta.Length; o++)
                            {
                                sum += layer.Weights[o, i] * delta[o];
                            }
                            nextError[i] = sum;
                        }
                        outputError = nextError;
                    }
                }
            }

            // --- GRADIENT CLIPPING ---
            // If norms exceed GradientClipNorm, we scale them down.
            ClipGradients(weightGrads, biasGrads, GradientClipNorm);

            // --- APPLY UPDATES ---
            lock (_lockObject)
            {
                for (int l = 0; l < _layers.Count; l++)
                {
                    int outSize = _layers[l].Biases.Length;
                    int inSize  = _layers[l].Inputs.Length;

                    for (int o = 0; o < outSize; o++)
                    {
                        float avgBiasGrad = biasGrads[l][o] / currentBatchSize;
                        _layers[l].Biases[o] -= learningRate * avgBiasGrad;

                        for (int i = 0; i < inSize; i++)
                        {
                            float avgWeightGrad = weightGrads[l][o][i] / currentBatchSize;
                            _layers[l].Weights[o, i] -= learningRate * avgWeightGrad;
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Clips gradients by global L2 norm if it exceeds maxNorm.
    /// </summary>
    private void ClipGradients(float[][][] weightGrads, float[][] biasGrads, float maxNorm)
    {
        // 1. Compute total L2 norm across all layers
        double totalSq = 0.0;
        for (int l = 0; l < _layers.Count; l++)
        {
            int outSize = weightGrads[l].Length;
            for (int o = 0; o < outSize; o++)
            {
                // biases
                double b = biasGrads[l][o];
                totalSq += b * b;

                // weights
                for (int i = 0; i < weightGrads[l][o].Length; i++)
                {
                    double w = weightGrads[l][o][i];
                    totalSq += w * w;
                }
            }
        }

        double norm = Math.Sqrt(totalSq);
        if (norm > maxNorm)
        {
            // scale factor
            float scale = (float)(maxNorm / norm);
            for (int l = 0; l < _layers.Count; l++)
            {
                int outSize = weightGrads[l].Length;
                for (int o = 0; o < outSize; o++)
                {
                    biasGrads[l][o] *= scale;
                    for (int i = 0; i < weightGrads[l][o].Length; i++)
                    {
                        weightGrads[l][o][i] *= scale;
                    }
                }
            }
        }
    }

    #endregion

    #region Activations

    private float[] ApplyActivation(float[] values, ActivationType act)
    {
        switch (act)
        {
            case ActivationType.Sigmoid:
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = 1f / (1f + (float)Math.Exp(-values[i]));
                    if (float.IsNaN(values[i]) || float.IsInfinity(values[i])) 
                        values[i] = 0f; // fallback
                }
                return values;

            case ActivationType.ReLU:
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = values[i] > 0 ? values[i] : 0f;
                    if (float.IsNaN(values[i]) || float.IsInfinity(values[i])) 
                        values[i] = 0f; 
                }
                return values;

            case ActivationType.Tanh:
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = (float)Math.Tanh(values[i]);
                    if (float.IsNaN(values[i]) || float.IsInfinity(values[i])) 
                        values[i] = 0f;
                }
                return values;

            case ActivationType.Softmax:
            {
                // 1) Find max for numerical stability
                float maxVal = float.NegativeInfinity;
                for (int i = 0; i < values.Length; i++)
                    if (values[i] > maxVal) maxVal = values[i];

                // 2) Exponentiate
                double sumExp = 0.0;
                for (int i = 0; i < values.Length; i++)
                {
                    double expVal = Math.Exp(values[i] - maxVal);
                    values[i] = (float)expVal;
                    sumExp += expVal;
                }

                // 3) Divide by sumExp
                if (sumExp < 1e-15) sumExp = 1e-15; // safety check
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] /= (float)sumExp;
                    // Clip to [SoftmaxEpsilon, 1 - SoftmaxEpsilon]
                    values[i] = Mathf.Clamp(values[i], SoftmaxEpsilon, 1f - SoftmaxEpsilon);

                    if (float.IsNaN(values[i]) || float.IsInfinity(values[i]))
                    {
                        values[i] = 1f / values.Length; // fallback (uniform)
                    }
                }
                return values;
            }

            default:
                throw new NotImplementedException($"Activation {act} not implemented.");
        }
    }

    private float DerivativeActivate(float activatedValue, ActivationType act)
    {
        switch (act)
        {
            case ActivationType.Sigmoid:
                // derivative is y*(1-y)
                return activatedValue * (1f - activatedValue);

            case ActivationType.ReLU:
                // derivative is 1 if y>0 else 0
                return (activatedValue > 0f) ? 1f : 0f;

            case ActivationType.Tanh:
                // derivative is 1 - y^2
                return 1f - activatedValue * activatedValue;

            case ActivationType.Softmax:
                // For cross-entropy + softmax, derivative wrt logits is (output[i] - target[i]).
                // We do a fallback derivative for MSE, etc.: y*(1-y).
                // However, in practice, with Softmax + CrossEntropy we rarely need this derivative in the same manner.
                return activatedValue * (1f - activatedValue);

            default:
                throw new NotImplementedException($"Derivative for {act} not implemented.");
        }
    }

    #endregion

    #region Loss

    private float[] ComputeOutputError(float[] output, float[] target, LossType loss, ActivationType outputAct)
    {
        // We typically do (output - target) for both MSE or CrossEntropy
        float[] error = new float[output.Length];
        switch (loss)
        {
            case LossType.MSE:
                for (int i = 0; i < output.Length; i++)
                {
                    error[i] = output[i] - target[i];
                    if (float.IsNaN(error[i]) || float.IsInfinity(error[i]))
                        error[i] = 0f;
                }
                break;

            case LossType.CrossEntropy:
                // Standard simplified approach: (output[i] - target[i]).
                // If using Softmax output, this aligns with the derivative wrt. logits.
                for (int i = 0; i < output.Length; i++)
                {
                    error[i] = output[i] - target[i];
                    if (float.IsNaN(error[i]) || float.IsInfinity(error[i]))
                        error[i] = 0f;
                }
                break;

            default:
                throw new NotImplementedException($"Loss {loss} not implemented.");
        }
        return error;
    }

    #endregion

    #region Data Shuffling

    private void ShuffleData(float[][] inputs, float[][] targets)
    {
        for (int i = 0; i < inputs.Length - 1; i++)
        {
            int j = _random.Next(i, inputs.Length);

            var tempIn = inputs[i];
            inputs[i] = inputs[j];
            inputs[j] = tempIn;

            var tempTg = targets[i];
            targets[i] = targets[j];
            targets[j] = tempTg;
        }
    }

    #endregion

    #region Save / Load

    [Serializable]
    private class SerializableNetwork
    {
        public SerializableLayer[] Layers;
        public ActivationType HiddenActivation;
        public ActivationType OutputActivation;
        public LossType Loss;
    }

    [Serializable]
    private class SerializableLayer
    {
        public float[] FlattenedWeights;  // Flattened 2D array
        public int WeightRows;            // Original dimensions
        public int WeightCols;
        public float[] Biases;
    }

    public void Save(string path)
    {
        lock (_lockObject)
        {
            var data = new SerializableNetwork
            {
                Layers = new SerializableLayer[_layers.Count],
                HiddenActivation = _hiddenActivation,
                OutputActivation = _outputActivation,
                Loss = _loss
            };

            // Copy weights and biases
            for (int i = 0; i < _layers.Count; i++)
            {
                var layer = _layers[i];
                int rows = layer.Weights.GetLength(0);
                int cols = layer.Weights.GetLength(1);
                
                // Flatten the 2D weights array
                float[] flatWeights = new float[rows * cols];
                for (int r = 0; r < rows; r++)
                {
                    for (int c = 0; c < cols; c++)
                    {
                        flatWeights[r * cols + c] = layer.Weights[r, c];
                    }
                }

                data.Layers[i] = new SerializableLayer
                {
                    FlattenedWeights = flatWeights,
                    WeightRows = rows,
                    WeightCols = cols,
                    Biases = layer.Biases
                };
            }

            string json = JsonUtility.ToJson(data);
            File.WriteAllText(path, json);
        }
    }

    public static NeuralNetwork Load(string path)
    {
        string json = File.ReadAllText(path);
        var data = JsonUtility.FromJson<SerializableNetwork>(json);

        // Calculate layer sizes
        int[] layerSizes = new int[data.Layers.Length + 1];
        layerSizes[0] = data.Layers[0].WeightCols;  // Input size
        for (int i = 0; i < data.Layers.Length; i++)
        {
            layerSizes[i + 1] = data.Layers[i].WeightRows;
        }

        // Create new network
        var nn = new NeuralNetwork(
            layerSizes,
            data.HiddenActivation,
            data.OutputActivation,
            data.Loss
        );

        // Restore weights and biases
        for (int i = 0; i < nn._layers.Count; i++)
        {
            var serializedLayer = data.Layers[i];
            var layer = nn._layers[i];
            
            // Reconstruct 2D weights array
            int rows = serializedLayer.WeightRows;
            int cols = serializedLayer.WeightCols;
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    layer.Weights[r, c] = serializedLayer.FlattenedWeights[r * cols + c];
                }
            }

            layer.Biases = serializedLayer.Biases;
        }

        return nn;
    }

    #endregion
}
