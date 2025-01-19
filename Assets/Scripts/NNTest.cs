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
        shape = new int[] { 2, 16, 16, 4 };
        nn = new NeuralNetwork(shape, NeuralNetwork.ReLU, NeuralNetwork.Softmax, 
            NeuralNetwork.ReLUDerivative, NeuralNetwork.SoftmaxDerivative, 
            NeuralNetwork.CategoricalCrossEntropy, NeuralNetwork.CategoricalCrossEntropyDerivative);
    }

    // Update is called once per frame
    void Update()
    {
        try {
            if (counter < maxIterations)
            {
                for (int i = 0; i < numIterationsPerUpdate; i++)
                {
                    List<float[]> inputs = new List<float[]>();
                    List<float[]> outputs = new List<float[]>();

                    // Generate training data
                    for (int j = 0; j < 256; j++)
                    {
                        float x = UnityEngine.Random.Range(-1f, 1f);
                        float y = UnityEngine.Random.Range(-1f, 1f);
                        
                        // Complex spiral pattern with 4 classes
                        float angle = Mathf.Atan2(y, x);
                        float radius = Mathf.Sqrt(x * x + y * y);
                        float adjustedAngle = angle + radius * 4f; // Creates spiral effect
                        int category = Mathf.FloorToInt(((adjustedAngle + Mathf.PI) / (2f * Mathf.PI) * 4f)) % 4;

                        inputs.Add(new float[] { x, y });
                        outputs.Add(new float[] {
                            category == 0 ? 1f : 0f,
                            category == 1 ? 1f : 0f,
                            category == 2 ? 1f : 0f,
                            category == 3 ? 1f : 0f
                        });
                    }

                    float avgCost = nn.TrainOneEpoch(inputs, outputs, learningRate, 32);
                    if (text != null) {
                        text.text = $"Iteration: {counter}, Cost: {avgCost:F6}";
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
