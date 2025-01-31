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
       int[] layers = new int[] { 2, 16, 16, 3 };
        nn = new NeuralNetwork(
            layers,
            ActivationType.ReLU,
            ActivationType.Softmax,
            ErrorType.CategoricalCrossEntropy
        );

        // nn.Load("Network");
            
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
            float totalLoss = 0;

            for (int i = 0; i < numIterationsPerUpdate; i++)
            {
                // Create jagged arrays instead of 2D arrays
                float[][] inputs = new float[32][];
                float[][] outputs = new float[32][];

                // Generate training data
                for (int j = 0; j < 32; j++)
                {
                    float x = UnityEngine.Random.Range(-1f, 1f);
                    float y = UnityEngine.Random.Range(-1f, 1f);

                    bool inSmallSpiral = PointInSpiral(x, y, 0.1f);
                    bool inBigSpiral = PointInSpiral(x, y, 0.14f);
                    
                    // Initialize the arrays for this batch item
                    inputs[j] = new float[2];
                    outputs[j] = new float[3];

                    // Set inputs
                    inputs[j][0] = x;
                    inputs[j][1] = y;

                    // Set outputs
                    if (inSmallSpiral) {
                        outputs[j] = new float[] {0, 1, 0};
                    } else if (inBigSpiral) {
                        outputs[j] = new float[] {1, 0, 0};
                    } else {
                        outputs[j] = new float[] {0, 0, 1};
                    }
                    
                }
                nn.Train(inputs, outputs, batchSize: 16, learningRate: 0.01f);


                counter += 1;
            }

            if (generationText != null) {
                generationText.text = $"Iteration: {counter}, Loss: {totalLoss/(numIterationsPerUpdate):F6}";
            }

        }
        nn.Save("network");

        for (int i = 0; i < quadResolution; i++) {
            for (int j = 0; j < quadResolution; j++) {
                float x = (i+0.5f) / (float)quadResolution - 0.5f;
                float y = (j+0.5f) / (float)quadResolution - 0.5f;
                float res = 1.0f / (float)quadResolution;
                
                float[] prediction = (float[])nn.Forward(new float[] {x*2.0f, y*2.0f});
                Material material = quadMaterials[i, j];

                // prediction[0] = certainty that it's red
                material.color = new Color((float)prediction[0], (float)prediction[1], (float)prediction[2]);
            }
        }
    }

    bool PointInSpiral(float x, float y, float spiralThickness) {        
        // Number of full rotations the spiral makes (0 -> maxRadius).
        int turns = 2;

        float r = Mathf.Sqrt(x*x + y*y);
        float theta = Mathf.Atan2(y, x);
        if (theta < 0) theta += 2.0f * Mathf.PI;

        // Maximum angle for 'turns' windings
        float maxTheta = turns * 2.0f * Mathf.PI;
        
        // If our point's angle is beyond the spiral's definition, it's out.
        if (theta > maxTheta) return false;

        // Linear "Archimedean" spiral from r=0 at theta=0 up to r=1 at theta=maxTheta
        float rIdeal = (theta / maxTheta) * 1.0f;  

        // Check distance from the spiral
        for (int i = 0; i < turns; i++) {
            if (Mathf.Abs(r - (rIdeal + (2.0f*Mathf.PI/maxTheta)*i)) <= spiralThickness) {
                return true;
            }
        }
        return false;
    }
}
