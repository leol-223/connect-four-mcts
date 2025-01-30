using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using System;

public class NNConvTest : MonoBehaviour
{
    public float learningRate = 0.001f;
    public TMPro.TMP_Text gamesDisplay;
    public TMPro.TMP_Text secondaryDisplay;
    public TMPro.TMP_Text generationText;
    public TMPro.TMP_Text thirdText;
    public TMPro.TMP_Text statsText;
    private NeuralNetwork nn;
    private List<float[]> trainingInputs;
    private List<float[]> trainingOutputs;
    private int batchSize = 32;
    private float targetCost = 0.01f; // Stop training when we reach this cost
    private int maxEpochs = 1000;

    private Material[,] quadMaterials;

    int counter = 0;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        // Create network with just error functions
        thirdText.text = "";
        statsText.text = "";
        gamesDisplay.text = "";
        secondaryDisplay.text = "";
        generationText.text = "";
        nn = new NeuralNetwork(NeuralNetwork.MSE, NeuralNetwork.MSEDerivative);

        // Add layers with their specific activation functions
        nn.AddConvolutionalLayer(
            inputDepth: 1,
            inputHeight: 8,
            inputWidth: 3,
            numFilters: 1,
            filterSize: 3,
            stride: 1,
            usePadding: false,
            activation: Activation.LeakyReLU
        );
        
        nn.AddPoolingLayer(
            inputDepth: 1,
            inputHeight: 6,
            inputWidth: 1,
            poolSize: 1,
            stride: 1
        );

        nn.AddDenseLayer(
            inputSize: 1 * 1 * 6,
            outputSize: 1,
            activation: Activation.Linear
        );

        GenerateTrainingData();
        DisplayStatus("Press Space to start training");
    }

    void GenerateTrainingData()
    {
        trainingInputs = new List<float[]>();
        trainingOutputs = new List<float[]>();

        // Generate training examples
        for (int n = 0; n < 1000; n++) // Increased dataset size
        {
            float[] input = new float[3 * 8];
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = -1.0f;
            }

            bool createPositiveExample = UnityEngine.Random.value > 0.5f;

            if (createPositiveExample)
            {
                int row = UnityEngine.Random.Range(0, 8);
                input[row * 3 + 0] = 1.0f;
                input[row * 3 + 1] = 1.0f;
                input[row * 3 + 2] = 1.0f;
                trainingOutputs.Add(new float[] { 1.0f });
            }
            else
            {
                int row1 = UnityEngine.Random.Range(0, 8);
                int row2 = UnityEngine.Random.Range(0, 8);
                int row3 = UnityEngine.Random.Range(0, 8);
                
                input[row1 * 3 + 0] = 1.0f;
                input[row2 * 3 + 2] = 1.0f;
                input[row3 * 3 + 1] = 1.0f;
                
                trainingOutputs.Add(new float[] { -1.0f });
            }

            trainingInputs.Add(input);
        }
    }

    void DisplayStatus(string status)
    {
        generationText.text = status;
    }

    void RunTests()
    {
        // Test cases
        float[] test1 = new float[3 * 8];
        for (int i = 0; i < test1.Length; i++) test1[i] = -1.0f;
        test1[0] = 1.0f; test1[1] = 1.0f; test1[2] = 1.0f; // Should output close to 1.0

        float[] test2 = new float[3 * 8];
        for (int i = 0; i < test2.Length; i++) test2[i] = -1.0f;
        test2[3] = 1.0f; test2[7] = 1.0f; test2[11] = 1.0f; // Should output close to -1.0

        float output1 = nn.Evaluate(test1)[0];
        float output2 = nn.Evaluate(test2)[0];

        string testResults = $"Test Results:\n" +
                           $"Horizontal line: {output1:F3} (expect ~1.0)\n" +
                           $"Random pattern: {output2:F3} (expect ~-1.0)";
        
        secondaryDisplay.text = testResults;
    }

    void Update()
    {
        try
        {
            if (Input.GetKeyDown(KeyCode.Space))
            {
                StartCoroutine(TrainNetwork());
            }

            if (Input.GetKeyDown(KeyCode.S))
            {
                nn.SaveNetwork("test.json");
                Debug.Log("Network saved successfully");
            }

            if (Input.GetKeyDown(KeyCode.L))
            {
                nn.LoadNetwork("test.json");
                Debug.Log("Network loaded successfully");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error in Update: {e.Message}\n{e.StackTrace}");
            DisplayStatus("Error occurred. Check console.");
            enabled = false;
        }
    }

    IEnumerator TrainNetwork()
    {
        DisplayStatus("Training started...");
        float currentCost = float.MaxValue;
        int epoch = 0;

        while (currentCost > targetCost && epoch < maxEpochs)
        {
            currentCost = nn.TrainOneEpoch(trainingInputs, trainingOutputs, learningRate, batchSize);
            
            if (float.IsNaN(currentCost))
            {
                DisplayStatus("Training failed - Cost is NaN");
                yield break;
            }

            if (epoch % 10 == 0)
            {
                DisplayStatus($"Epoch {epoch}\nCost: {currentCost:F4}\nTarget: {targetCost:F4}");
                RunTests();
                yield return null; // Let the UI update
            }
            
            epoch++;
        }

        string finalStatus = currentCost <= targetCost 
            ? $"Training succeeded!\nFinal cost: {currentCost:F4}\nEpochs: {epoch}" 
            : $"Training stopped.\nFinal cost: {currentCost:F4}\nEpochs: {epoch}";
        
        DisplayStatus(finalStatus);
        RunTests();
    }
}
