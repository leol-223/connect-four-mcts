using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using System;
using System.Linq;


public class NNTest : MonoBehaviour
{
    public TMPro.TMP_Text gamesDisplay;
    public TMPro.TMP_Text secondaryDisplay;
    public TMPro.TMP_Text generationText;
    public TMPro.TMP_Text thirdText;
    public TMPro.TMP_Text statsText;
    public int numIterationsPerUpdate;
    public int maxIterations;
    public int quadResolution;
    public Color activeColor;
    public Color inactiveColor;

    NeuralNetwork nn;
    private bool isSaved = false;

    private Material[,] quadMaterials;

    int counter = 0;
    
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        // Increase network capacity slightly
        int[] layers = new int[] { 2, 32, 32, 2 };
        nn = new NeuralNetwork(
            layers,
            NeuralNetwork.ActivationType.ReLU,
            NeuralNetwork.ActivationType.Softmax,
            NeuralNetwork.LossType.CrossEntropy
        );

        nn = NeuralNetwork.Load("network");
            
        gamesDisplay.text = "";
        secondaryDisplay.text = "";
        generationText.text = "";
        thirdText.text = "";
        statsText.text = "";

        quadMaterials = new Material[quadResolution, quadResolution];

        for (int i = 0; i < quadResolution; i++) {
            for (int j = 0; j < quadResolution; j++) {
                float x = (i+0.5f) / (float)quadResolution - 0.5f;
                float y = (j+0.5f) / (float)quadResolution - 0.5f;
                float res = 1f / (float)quadResolution;
                // Create a quad GameObject
                GameObject quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
                quad.transform.position = new Vector3(x*7, y*7-1, 1);  // Slightly in front of camera
                quad.transform.localScale = new Vector3(res*7, res*7, 1);
                
                // Create an unlit material so it's always visible
                Material material = new Material(Shader.Find("Unlit/Color"));
                material.color = new Color(0.5f, 0.5f, 0.5f);
                quad.GetComponent<MeshRenderer>().material = material;
                quadMaterials[i, j] = material;
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (counter < maxIterations)
        {
            for (int i = 0; i < numIterationsPerUpdate; i++)
            {
                float[][] inputs = new float[128][]; // Increased batch size
                float[][] outputs = new float[128][];

                // Generate training data
                for (int j = 0; j < 128; j++)
                {
                    float x = UnityEngine.Random.Range(-1f, 1f);
                    float y = UnityEngine.Random.Range(-1f, 1f);

                    bool insideCircle = IsInsideCircle(x, y, 0.8f);
                    
                    inputs[j] = new float[2];
                    outputs[j] = new float[2];

                    inputs[j][0] = x;
                    inputs[j][1] = y;

                    // Make the target values less extreme
                    outputs[j] = insideCircle ? new float[] {1f, 0f} : new float[] {0f, 1f};
                }
                
                // Increased learning rate and batch size
                nn.TrainEpoch(inputs, outputs, batchSize: 64, learningRate: 0.001f);
                counter += 1;
            }

            // Test loss calculation
            float totalLoss = 0;
            int testSamples = 1000;
            for (int j = 0; j < testSamples; j++) 
            {
                float x = UnityEngine.Random.Range(-1f, 1f);
                float y = UnityEngine.Random.Range(-1f, 1f);

                bool insideCircle = IsInsideCircle(x, y, 0.5f);
                
                float[] input_test = new float[] { x, y };
                float[] output_test = insideCircle ? new float[] {1, 0} : new float[] {0, 1};

                float[] prediction = nn.Forward(input_test);
                totalLoss += CalculateCrossEntropyLoss(prediction, output_test);
            }
            nn.Save("network");

            if (generationText != null) {
                generationText.text = $"Iteration: {counter}, Loss: {totalLoss/testSamples:F6}";
            }
        }

        // Visualization update
        for (int i = 0; i < quadResolution; i++) {
            for (int j = 0; j < quadResolution; j++) {
                float x = (i+0.5f) / (float)quadResolution - 0.5f;
                float y = (j+0.5f) / (float)quadResolution - 0.5f;
                
                float[] prediction = nn.Forward(new float[] {x*2.0f, y*2.0f});
                Material material = quadMaterials[i, j];

                // Red for inside circle, blue for outside
                material.color = new Color(prediction[0], 0, prediction[1]);
            }
        }
    }

    private bool IsInsideCircle(float x, float y, float radius)
    {
        return x*x + y*y <= radius*radius;
    }

    public float CalculateLoss(float[] arr1, float[] arr2) {
        float tot = 0;
        for (int i = 0; i < arr1.Length; i++) {
            tot += (arr1[i] - arr2[i]) * (arr1[i] - arr2[i]);
        }
        return tot;
    }

    public float CalculateCrossEntropyLoss(float[] predicted, float[] actual) {
        float loss = 0;
        for (int i = 0; i < predicted.Length; i++) {
            // Add small epsilon to prevent log(0)
            float clippedPred = Mathf.Clamp(predicted[i], 1e-7f, 1 - 1e-7f);
            loss += -actual[i] * Mathf.Log(clippedPred);
        }
        return loss;
    }
}
