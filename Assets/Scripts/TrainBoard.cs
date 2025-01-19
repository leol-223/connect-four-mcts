using UnityEngine;
using System.Collections.Generic;
using System;
using System.Threading.Tasks;
using System.Threading;
using System.Linq;

public class TrainBoard : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    public int[] valueShape;
    public int[] policyShape;
    public NeuralNetwork valueNetwork;
    public NeuralNetwork policyNetwork;
    public int numGenerations;
    public int startGeneration;
    public int numGames;
    public int numEpochs;
    public int initialIterations;
    public float iterationsGrowthRate;
    public int maxIterations;
    public string modelName;
    public float initialTemperature;
    public float temperatureDecay;
    public float initialRootNoise;
    public float rootNoiseDecay;
    public float initialDirichletAlpha;
    public float dirichletAlphaDecay;
    public float trainRatio;
    public float initialLearningRate = 0.001f;
    public float learningRateDecay = 0.95f;

    public TMPro.TMP_Text gamesDisplay;
    public TMPro.TMP_Text secondaryDisplay;
    public TMPro.TMP_Text generationText;
    public TMPro.TMP_Text thirdText;

    private List<float[]> valueInputs;
    private List<float[]> valueOutputs;
    private List<float[]> policyInputs;
    private List<float[]> policyOutputs;

    private List<float[]> valueTrainInputs;
    private List<float[]> valueTrainOutputs;
    private List<float[]> valueTestInputs;
    private List<float[]> valueTestOutputs;
    private List<float[]> policyTrainInputs;
    private List<float[]> policyTrainOutputs;
    private List<float[]> policyTestInputs;
    private List<float[]> policyTestOutputs;
    private HashSet<BoardState> uniquePositions;

    private int epochCounter = 0;
    private int counter = 0;
    private int generationCounter = 0;
    private float startTime;

    // Add this struct at the class level
    private struct BoardState : IEquatable<BoardState>
    {
        public ulong redBoard;
        public ulong yellowBoard;

        public BoardState(ulong red, ulong yellow)
        {
            redBoard = red;
            yellowBoard = yellow;
        }

        public override bool Equals(object obj)
        {
            if (obj is BoardState other)
                return Equals(other);
            return false;
        }

        public bool Equals(BoardState other)
        {
            return redBoard == other.redBoard && yellowBoard == other.yellowBoard;
        }

        public override int GetHashCode()
        {
            return redBoard.GetHashCode() ^ yellowBoard.GetHashCode();
        }
    }

    void Start()
    {
        valueNetwork = new NeuralNetwork(valueShape, NeuralNetwork.ReLU, NeuralNetwork.Tanh, NeuralNetwork.ReLUDerivative, NeuralNetwork.TanhDerivative, NeuralNetwork.MSE, NeuralNetwork.MSEDerivative);
        policyNetwork = new NeuralNetwork(policyShape, NeuralNetwork.ReLU, NeuralNetwork.Softmax, NeuralNetwork.ReLUDerivative, NeuralNetwork.SoftmaxDerivative, NeuralNetwork.CategoricalCrossEntropy, NeuralNetwork.CategoricalCrossEntropyDerivative);

        if (startGeneration > 0) {
            valueNetwork.LoadNetwork(modelName+"-V" + startGeneration.ToString());
            policyNetwork.LoadNetwork(modelName+"-P" + startGeneration.ToString());
            generationCounter = startGeneration;
        }

        valueInputs = new List<float[]>();
        valueOutputs = new List<float[]>();

        policyInputs = new List<float[]>();
        policyOutputs = new List<float[]>();

        startTime = Time.realtimeSinceStartup;
        uniquePositions = new HashSet<BoardState>();
    }

    // Update is called once per frame
    void Update()
    {
        if (generationCounter < numGenerations)
        {
            generationText.text = "Generation " + (generationCounter + 1).ToString();
            if (counter < numGames)
            {
                int currentIterations = Mathf.Min(
                    maxIterations,
                    Mathf.RoundToInt(initialIterations * Mathf.Pow(iterationsGrowthRate, generationCounter))
                );

                float temperature = initialTemperature * Mathf.Pow(temperatureDecay, generationCounter);

                Debug.Log($"Temperature: {temperature}");
                Debug.Log($"Num iterations: {currentIterations}");

                secondaryDisplay.text = "Epoch 0: -";

                float currentNoise = initialRootNoise * Mathf.Pow(rootNoiseDecay, generationCounter);
                float currentAlpha = initialDirichletAlpha * Mathf.Pow(dirichletAlphaDecay, generationCounter);
                
                BoardNN board = new BoardNN();
                board.valueNetwork = valueNetwork;
                board.policyNetwork = policyNetwork;
                board.rootNoise = currentNoise;
                board.dirichletAlpha = currentAlpha;

                List<float[]> positions = new List<float[]>();
                BoardNN.Player winningPlayer;
                BoardNN.Player currentPlayer = BoardNN.Player.Red;

                while (true)
                {                
                    winningPlayer = board.GetWinningPlayer();
                    if (winningPlayer != BoardNN.Player.None || board.IsFull())
                    {
                        break;
                    }

                    float[] position = GetPosition(board.redBitboard, board.yellowBitboard);
                    float[] symmetricalPosition = GetSymmetricalPosition(board.redBitboard, board.yellowBitboard);
                    uniquePositions.Add(new BoardState(board.redBitboard, board.yellowBitboard));

                    int move;
                    if (currentPlayer == BoardNN.Player.Red)
                    {
                        BoardNN.MoveEval bestMove = board.TreeSearch(currentIterations, true, temperature);
                        board.MakeMove(bestMove.Move, currentPlayer);
                        move = bestMove.Move;
                        currentPlayer = BoardNN.Player.Yellow;
                    }
                    else
                    {
                        BoardNN.MoveEval bestMove = board.TreeSearch(currentIterations, false, temperature);
                        board.MakeMove(bestMove.Move, currentPlayer);
                        move = bestMove.Move;
                        currentPlayer = BoardNN.Player.Red;
                    }
                    
                    positions.Add(position);
                    if (position != symmetricalPosition) {
                        positions.Add(symmetricalPosition);
                    }
                    
                    float[] promise = new float[7];
                    
                    // Normalize promises and explicitly zero out illegal moves
                    float sum = 0;
                    for (int i = 0; i < 7; i++) {
                        promise[i] = board.promise[i];
                    }
                    
                    policyInputs.Add(position);
                    policyOutputs.Add(promise);
                    policyInputs.Add(symmetricalPosition);
                    policyOutputs.Add(promise.Reverse().ToArray());
                }

                float[] output;
                if (winningPlayer == BoardNN.Player.Red)
                {
                    output = new float[1] { 1};
                }
                else if (winningPlayer == BoardNN.Player.Yellow)
                {
                    output = new float[1] { -1f};
                }
                else
                {
                    output = new float[1] {0f};
                }
                Debug.Log(output[0]);
                for (int i = 0; i < positions.Count; i++)
                {
                    valueInputs.Add(positions[i]);
                    valueOutputs.Add(output);
                }

                counter += 1;
                gamesDisplay.text = counter.ToString() + "/" + numGames.ToString() + " Games";
                float currentTime = Time.realtimeSinceStartup;
                float timePerGame = (currentTime - startTime) / counter;
                float timeLeft = timePerGame * (numGames - counter);
                secondaryDisplay.text = " Time Left: ~" + Mathf.RoundToInt(timeLeft).ToString() + "s";

                thirdText.text = $"{uniquePositions.Count} Unique Positions";
            }
            else if (epochCounter < numEpochs)
            {
                int trainTestSplit = (int)(trainRatio * valueInputs.Count);

                // Using GetRange for Lists
                valueTrainInputs = valueInputs.GetRange(0, trainTestSplit);
                valueTestInputs = valueInputs.GetRange(trainTestSplit, valueInputs.Count - trainTestSplit);

                valueTrainOutputs = valueOutputs.GetRange(0, trainTestSplit);
                valueTestOutputs = valueOutputs.GetRange(trainTestSplit, valueOutputs.Count - trainTestSplit);

                policyTrainInputs = policyInputs.GetRange(0, trainTestSplit);
                policyTestInputs = policyInputs.GetRange(trainTestSplit, policyInputs.Count - trainTestSplit);

                policyTrainOutputs = policyOutputs.GetRange(0, trainTestSplit);
                policyTestOutputs = policyOutputs.GetRange(trainTestSplit, policyOutputs.Count - trainTestSplit);

                Debug.Log($"Training data length: {valueTrainInputs.Count}, Testing data length: {valueTestInputs.Count}");
                
                float currentLearningRate = initialLearningRate * Mathf.Pow(learningRateDecay, epochCounter);

                valueNetwork.TrainOneEpoch(valueTrainInputs, valueTrainOutputs, currentLearningRate, 128);
                policyNetwork.TrainOneEpoch(policyTrainInputs, policyTrainOutputs, currentLearningRate * 2f, 128);

                float cost1 = valueNetwork.CalculateCost(valueTestInputs, valueTestOutputs);
                float cost2 = policyNetwork.CalculateCost(policyTestInputs, policyTestOutputs);
                
                epochCounter += 1;
                Debug.Log("Value cost: " + cost1.ToString());
                Debug.Log("Policy cost: " + cost2.ToString());
                secondaryDisplay.text = "Epoch " + epochCounter.ToString() + " | Value cost: " + cost1.ToString("F4") + " | Policy cost: " + cost2.ToString("F4");
            }
            else
            {
                generationCounter += 1;
                valueNetwork.SaveNetwork(modelName+"-V" + generationCounter.ToString());
                policyNetwork.SaveNetwork(modelName+"-P" + generationCounter.ToString());

                valueInputs.Clear();
                valueOutputs.Clear();
                policyInputs.Clear();
                policyOutputs.Clear();
                uniquePositions.Clear();
                
                epochCounter = 0;
                counter = 0;
                startTime = Time.realtimeSinceStartup;
            }
        }
    }

    public static float[] GetPosition(ulong redBitboard, ulong yellowBitboard)
    {
        // Create the output array for 84 floats (42 for red, 42 for yellow)
        float[] result = new float[84];

        // Match the exact same position calculation as ShowBoard
        for (int col = 0; col < 7; col++)
        {
            for (int row = 0; row < 6; row++)
            {
                int bitPosition = 8 * col + row;
                int nnPosition = col * 6 + row;
                
                bool isRed = (redBitboard & ((ulong)1 << bitPosition)) != 0;
                bool isYellow = (yellowBitboard & ((ulong)1 << bitPosition)) != 0;
                
                result[nnPosition] = isRed ? 1.0f : 0.0f;
                result[nnPosition + 42] = isYellow ? 1.0f : 0.0f;
            }
        }

        return result;
    }

    public float[] GetSymmetricalPosition(ulong redBitboard, ulong yellowBitboard)
    {
        // First get the neural network input representation
        float[] nnInput = GetPosition(redBitboard, yellowBitboard);
        float[] symmetricInput = new float[84];

        // For each row
        for (int row = 0; row < 6; row++)
        {
            // For each column
            for (int col = 0; col < 7; col++)
            {
                // Calculate source and target indices
                int sourceIndex = col * 6 + row;
                int targetIndex = (6 - col) * 6 + row;  // Flip column horizontally

                // Copy red pieces (first 42 values)
                symmetricInput[targetIndex] = nnInput[sourceIndex];
                
                // Copy yellow pieces (last 42 values)
                symmetricInput[targetIndex + 42] = nnInput[sourceIndex + 42];
            }
        }

        return symmetricInput;
    }

    public static void ShowBoardFromNNInput(float[] nnInput)
    {
        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        sb.AppendLine("Board State from NN Input:");
        sb.AppendLine("-----------------------------");
        
        // Print from top to bottom
        for (int row = 5; row >= 0; row--)
        {
            string currentRow = "|";
            for (int col = 0; col < 7; col++)
            {
                // Calculate position in the NN input array
                // For each position, check both red and yellow arrays
                int index = (col * 6) + row;  // Position in the column
                bool isRed = nnInput[index] > 0.5f;
                bool isYellow = nnInput[index + 42] > 0.5f;
                
                char piece = ' ';
                if (isRed) piece = 'R';
                else if (isYellow) piece = 'Y';
                
                currentRow += $" {piece} |";
            }
            sb.AppendLine(currentRow);
        }
        
        sb.AppendLine("-----------------------------");
        sb.AppendLine("  1   2   3   4   5   6   7  ");
        
        Debug.Log(sb.ToString());
    }
}