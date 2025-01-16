using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using System;

public class NNTest : MonoBehaviour
{
    public int[] shape;
    public float learningRate;
    public TMPro.TMP_Text text;
    public int numIterationsPerUpdate;
    public int maxIterations;
    NeuralNetwork nn;
    private bool isSaved = false;

    int counter = 0;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        try {
            Debug.Log("Starting NNTest initialization...");
            Debug.Log($"Network shape: {string.Join(", ", shape)}");
            
            nn = new NeuralNetwork(shape, NeuralNetwork.ReLU, NeuralNetwork.Softmax, 
                NeuralNetwork.ReLUDerivative, NeuralNetwork.SoftmaxDerivative, 
                NeuralNetwork.CategoricalCrossEntropy, NeuralNetwork.CategoricalCrossEntropyDerivative);
            
            Debug.Log("Neural network initialized successfully");
        }
        catch (Exception e) {
            Debug.LogError($"Error in Start: {e.Message}\n{e.StackTrace}");
        }
    }

    // Update is called once per frame
    void Update()
    {
        try {
            if (nn == null) {
                Debug.LogError("Neural network is null!");
                return;
            }

            if (counter < maxIterations)
            {
                for (int i = 0; i < numIterationsPerUpdate; i++)
                {
                    List<float[]> inputs = new List<float[]>();
                    List<float[]> outputs = new List<float[]>();

                    // Generate training data
                    for (int j = 0; j < 128; j++)
                    {
                        float x = UnityEngine.Random.Range(-1f, 1f);
                        float y = UnityEngine.Random.Range(-1f, 1f);
                        bool inCircle = (x * x + y * y <= 1);

                        inputs.Add(new float[] { x, y });
                        outputs.Add(inCircle ? new float[] { 1, 0, 0 } : new float[] { 0, 1, 0 });
                    }

                    float avgCost = nn.TrainOneEpoch(inputs, outputs, learningRate, 15);
                    if (text != null) {
                        text.text = "Cost: " + avgCost.ToString();
                    }

                    counter += 1;
                }
            }
            else if (!isSaved)
            {
                nn.SaveNetwork("test.json");
                isSaved = true;
                Debug.Log("Network saved successfully");
            }
        }
        catch (Exception e) {
            Debug.LogError($"Error in Update: {e.Message}\n{e.StackTrace}");
            enabled = false; // Stop the Update loop if we hit an error
        }
    }
}
