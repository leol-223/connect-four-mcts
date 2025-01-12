using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class NNTest : MonoBehaviour
{
    public int[] shape;
    public float learningRate;
    public TMPro.TMP_Text text;
    public int numIterationsPerUpdate;
    public int maxIterations;
    NeuralNetwork nn;

    int counter = 0;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        nn = new NeuralNetwork(shape, NeuralNetwork.Tanh, NeuralNetwork.Softmax, NeuralNetwork.TanhDerivative, NeuralNetwork.SoftmaxDerivative, NeuralNetwork.MSE, NeuralNetwork.MSEDerivative);
    }

    // Update is called once per frame
    void Update()
    {
        if (counter < maxIterations)
        {
            for (int i = 0; i < numIterationsPerUpdate; i++)
            {
                List<float[]> inputs = new List<float[]>();
                List<float[]> outputs = new List<float[]>();

                for (int j = 0; j < 128; j++)
                {
                    float x = Random.Range(-1f, 1f);
                    float y = Random.Range(-1f, 1f);
                    bool inCircle = (x * x + y * y <= 1);

                    float[] input = new float[] { x, y };
                    float[] output;
                    if (inCircle)
                    {
                        output = new float[] { 1, 0 };
                    } else
                    {
                        output = new float[] { 0, 1 };
                    }
                    inputs.Add(input);
                    outputs.Add(output);
                }

                float avgCost = nn.TrainOneEpoch(inputs, outputs, learningRate, 16);
                text.text = "Cost: " + avgCost.ToString();

                counter += 1;
            }
        }
    }
}
