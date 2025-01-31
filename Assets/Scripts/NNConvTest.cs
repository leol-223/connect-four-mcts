using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using System;
using System.Linq;


public class NNConvTest : MonoBehaviour
{
    public TMPro.TMP_Text gamesDisplay;
    public TMPro.TMP_Text secondaryDisplay;
    public TMPro.TMP_Text generationText;
    public TMPro.TMP_Text thirdText;
    public TMPro.TMP_Text statsText;
    public int numIterationsPerUpdate;
    public int maxIterations;
    public Color activeColor;
    public Color inactiveColor;
    private Material[,] quadMaterials;

    NeuralNetwork nn;
    private bool isSaved = false;

    int counter = 0;
    /*
    
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        nn = new NeuralNetwork(LossType.CategoricalCrossEntropy);

        // 4 x 6
        nn.AddLayer(new ConvolutionalLayer(inChannels: 1, outChannels: 4, inHeight: 4, inWidth: 6, kernelHeight: 2, kernelWidth: 2, activation: ActivationType.LeakyReLU, stride: 1));
        nn.AddLayer(new ConvolutionalLayer(inChannels: 4, outChannels: 4, inHeight: 3, inWidth: 5, kernelHeight: 3, kernelWidth: 3, activation: ActivationType.LeakyReLU, stride: 1));
        nn.AddLayer(new FlattenLayer());
        // 1 x 3
        nn.AddLayer(new DenseLayer(4 * 3, 6, ActivationType.ReLU));  // 24 = 4*6 (input size)
        nn.AddLayer(new DenseLayer(6, 2, ActivationType.Softmax));

        // nn.Load("network");
            
        gamesDisplay.text = "";
        secondaryDisplay.text = "";
        generationText.text = "";
        thirdText.text = "";
        statsText.text = "";

        quadMaterials = new Material[6, 6];

        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                float y = (i+1f) / 6f - 0.5f;
                float x = (j+0.5f) / 6f - 0.5f;
                float res = 1f / (float)6;
                // Create a quad GameObject
                GameObject quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
                quad.transform.position = new Vector3(x*7, y*7-1, 1);  // Slightly in front of camera
                quad.transform.localScale = new Vector3(res*7, res*7, 1);
                
                // Create an unlit material so it's always visible
                Material material = new Material(Shader.Find("Unlit/Color"));
                material.color = new Color(1f, 1f, 1f);
                quad.GetComponent<MeshRenderer>().material = material;
                quadMaterials[i, j] = material;
            }
        }
    }

    void DisplayArr(float[] arr) {
        // Assuming arr is a flattened 1x4x6 array (channel x height x width)
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 6; j++) {
                // i -> row
                // j -> column
                Material quadMaterial = quadMaterials[i, j];
                
                // Since we have 1 channel, we can directly access the value
                // Index = channel_offset + row*width + col
                // With 1 channel: index = i*6 + j
                float value = arr[i*6 + j];
                
                // Update color based on value
                quadMaterial.color = value == 1f ? activeColor : inactiveColor;
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        // CheckGradients();
        if (counter < maxIterations)
        {
            float totalLoss = 0;

            for (int i = 0; i < numIterationsPerUpdate; i++)
            {
                 // Create jagged arrays instead of 2D arrays
                float[][] inputs = new float[32][];
                float[][] outputs = new float[32][];

                for (int j = 0; j < 32; j++)
                {
                    // channels, height, width
                    float[,,] arr = new float[1, 4, 6];
                    for (int row = 0; row < 4; row++) {
                        for (int col = 0; col < 6; col++) {
                            arr[0, row, col] = (float)UnityEngine.Random.Range(0, 2);
                        }
                    }

                    // check for horizontal lines
                    bool hasAny = false;
                    for (int row = 0; row < 4; row++) {
                        for (int column = 0; column < 6-3; column++) {
                            bool hasLine = true;
                            for (int increment = 0; increment < 4; increment++) {
                                if (arr[0, row, column+increment] == 0) {
                                    hasLine = false;
                                }
                            }
                            if (hasLine) {
                                hasAny = true;
                            }
                        }
                    }

                     // Create jagged arrays instead of 2D arrays
                    // Store in batch arrays
                    inputs[j] = NeuralNetwork.Flatten3DTo1D(arr);
                    outputs[j] = hasAny ? new float[] {1, 0} : new float[] {0, 1};
                }

                DisplayArr(inputs[31]);
                float[] label = outputs[31];
                float[] prediction = nn.Forward(inputs[31]);

                string predictionText;
                if (prediction[0] > prediction[1]) {
                    predictionText = $"Prediction: True ({prediction[0]*100:F2}%)";
                } else {
                    predictionText = $"Prediction: False ({prediction[1]*100:F2}%)";
                }

                if (label[0] == 1) {
                    statsText.text = "\nLabel: True\n" + predictionText;
                } else {
                    statsText.text = "\nLabel: False\n" + predictionText;
                }

                float[][] predictionsBatch = nn.ForwardBatch(inputs);

                // 2) Compute average loss
                float loss = nn.ComputeLossBatch(predictionsBatch, outputs);

                // 3) Backprop
                nn.BackwardBatch(predictionsBatch, outputs, 0.001f);
                totalLoss += loss;

                counter += 1;
            }

            if (generationText != null) {
                generationText.text = $"Iteration: {counter}, Loss: {totalLoss/(numIterationsPerUpdate):F6}";
            }

        }
        nn.Save("network");
    }
    */
}

