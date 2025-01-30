using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using System;

public class NNTest : MonoBehaviour
{
    public int[] shape;
    public float learningRate;
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
        nn = new NeuralNetwork(
            NeuralNetwork.CategoricalCrossEntropy, NeuralNetwork.CategoricalCrossEntropyDerivative);

        // Assuming shape is your array of layer sizes, e.g., [64, 32, 16, 4]
        for (int i = 0; i < shape.Length - 1; i++) {
            nn.AddDenseLayer(
                inputSize: shape[i],
                outputSize: shape[i + 1],
                activation: i == shape.Length - 2 ? Activation.Softmax : Activation.LeakyReLU
            );
        }
            
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
                material.color = Color.Lerp(inactiveColor, activeColor, 0.5f);
                quad.GetComponent<MeshRenderer>().material = material;
                quadMaterials[i, j] = material;
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        for (int i = 0; i < quadResolution; i++) {
            for (int j = 0; j < quadResolution; j++) {
                float x = (i+0.5f) / (float)quadResolution - 0.5f;
                float y = (j+0.5f) / (float)quadResolution - 0.5f;
                float res = 1f / (float)quadResolution;
                
                float[] prediction = nn.Evaluate(new float[] {x*2f, y*2f});
                Material material = quadMaterials[i, j];

                // prediction[0] = certainty that it's red
                material.color = Color.Lerp(inactiveColor, activeColor, prediction[0]);
            }
        }

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

                        bool isActive = PointInSpiral(x, y);
                        
                        outputs.Add(isActive ? new float[] {1, 0} : new float[] {0, 1});
                        inputs.Add(new float[] {x, y});
                    }

                    float avgCost = nn.TrainOneEpoch(inputs, outputs, learningRate, 32);
                    if (generationText != null) {
                        generationText.text = $"Iteration: {counter+1}, Cost: {avgCost:F6}";
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

    bool PointInSpiral(float x, float y) {
            // How "fat" the spiral is.
        float spiralThickness = 0.1f;
        
        // Number of full rotations the spiral makes (0 -> maxRadius).
        int turns = 2;

        float r = Mathf.Sqrt(x*x + y*y);
        float theta = Mathf.Atan2(y, x);
        if (theta < 0) theta += 2f * Mathf.PI;

        // Maximum angle for 'turns' windings
        float maxTheta = turns * 2f * Mathf.PI;
        
        // If our point's angle is beyond the spiral's definition, it's out.
        if (theta > maxTheta) return false;

        // Linear “Archimedean” spiral from r=0 at theta=0 up to r=1 at theta=maxTheta
        float rIdeal = (theta / maxTheta) * 1.0f;  

        // Check distance from the spiral
        for (int i = 0; i < turns; i++) {
            if (Mathf.Abs(r - (rIdeal + (2f*Mathf.PI/maxTheta)*i)) <= spiralThickness) {
                return true;
            }
        }
        return false;
    }
}
