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
    public int generationIncrement;
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
    public float generationLearningRateDecay = 0.98f;
    public float epochLearningRateDecay = 0.95f;
    public float minTemperature = 0.2f;

    public TMPro.TMP_Text gamesDisplay;
    public TMPro.TMP_Text secondaryDisplay;
    public TMPro.TMP_Text generationText;
    public TMPro.TMP_Text thirdText;
    public TMPro.TMP_Text statsText;

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
    private int[] numPositions;

    private int epochCounter = 0;
    private int counter = 0;
    private int generationCounter = 0;
    private float startTime;

    private int numRedWins;
    private int numYellowWins;
    private int numDraws;

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
        TreeNode.SetExplorationConstraints(false);

        numPositions = new int[42];
        for (int i = 0; i < 42; i++) {
            numPositions[i] = 0;
        }
        valueNetwork = new NeuralNetwork(valueShape, NeuralNetwork.ReLU, NeuralNetwork.Tanh, NeuralNetwork.ReLUDerivative, NeuralNetwork.TanhDerivative, NeuralNetwork.MSE, NeuralNetwork.MSEDerivative);
        policyNetwork = new NeuralNetwork(policyShape, NeuralNetwork.ReLU, NeuralNetwork.Softmax, NeuralNetwork.ReLUDerivative, NeuralNetwork.SoftmaxDerivative, NeuralNetwork.CategoricalCrossEntropy, NeuralNetwork.CategoricalCrossEntropyDerivative);

        if (startGeneration+generationIncrement > 0) {
            valueNetwork.LoadNetwork(modelName+"-V" + (startGeneration+generationIncrement).ToString());
            policyNetwork.LoadNetwork(modelName+"-P" + (startGeneration+generationIncrement).ToString());
            generationCounter = startGeneration;
        }

        valueInputs = new List<float[]>();
        valueOutputs = new List<float[]>();

        policyInputs = new List<float[]>();
        policyOutputs = new List<float[]>();

        startTime = Time.realtimeSinceStartup;
        uniquePositions = new HashSet<BoardState>();
        
        numRedWins = 0;
        numYellowWins = 0;
        numDraws = 0;
    }

    // Update is called once per frame
    void Update()
    {
        if (generationCounter < numGenerations)
        {
            generationText.text = "Generation " + (generationCounter + generationIncrement + 1).ToString();
            if (counter < numGames)
            {
                int currentIterations = Mathf.Min(
                    maxIterations,
                    Mathf.RoundToInt(initialIterations * Mathf.Pow(iterationsGrowthRate, generationCounter))
                );

                float temperature = Mathf.Max(
                    minTemperature,
                    initialTemperature * Mathf.Pow(temperatureDecay, generationCounter)
                );


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

                int numPositionLocal = 0;
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

                    // ShowBoardFromNNInput(position);
                    // Debug.Log(valueNetwork.Evaluate(position)[0]);

                    numPositionLocal += 1;
                }

                float[] output;
                if (winningPlayer == BoardNN.Player.Red)
                {
                    output = new float[1] { 1};
                    numRedWins += 1;
                }
                else if (winningPlayer == BoardNN.Player.Yellow)
                {
                    output = new float[1] { -1f};
                    numYellowWins += 1;
                }
                else
                {
                    output = new float[1] {0f};
                    numDraws += 1;
                }

                // Debug.Log(output[0]);

                for (int i = 0; i < positions.Count; i++)
                {
                    valueInputs.Add(positions[i]);
                    valueOutputs.Add(output);
                }

                numPositions[numPositionLocal-1] += 1;

                int minIndex = -1; 
                int maxIndex = -1;
                int totalGames = 0;
                int medianIndex = -1;

                // First pass to find min, max, and total
                for (int i = 0; i < 42; i++) {
                    if (numPositions[i] > 0 && minIndex == -1) {
                        minIndex = i;
                    }
                    if (numPositions[i] > 0) {
                        maxIndex = i;
                        totalGames += numPositions[i];
                    }
                }

                counter += 1;
                gamesDisplay.text = counter.ToString() + "/" + numGames.ToString() + " Games";
                float currentTime = Time.realtimeSinceStartup;
                float timePerGame = (currentTime - startTime) / counter;
                float timeLeft = timePerGame * (numGames - counter);
                secondaryDisplay.text = " Time Left: ~" + Mathf.RoundToInt(timeLeft).ToString() + "s";

                // Second pass to find median
                int cumSum = 0;
                int medianThreshold = totalGames / 2;
                for (int i = 0; i < 42; i++) {
                    cumSum += numPositions[i];
                    if (cumSum > medianThreshold && medianIndex == -1) {
                        medianIndex = i;
                    }
                }

                thirdText.text = $"{uniquePositions.Count} Unique Positions | Med: {medianIndex+1} | Range: {minIndex+1} - {maxIndex+1}";
                statsText.text = $"<color=#b31414>{numRedWins} red</color> | <color=#636262>{numDraws} draw</color> | <color=#959711>{numYellowWins} yellow</color>";
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
                
                // Base learning rate decays with generations
                float generationAdjustedLR = initialLearningRate * Mathf.Pow(generationLearningRateDecay, generationCounter);
                // Further decay within epochs
                float currentLearningRate = generationAdjustedLR * Mathf.Pow(epochLearningRateDecay, epochCounter);

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
                valueNetwork.SaveNetwork(modelName+"-V" + (generationCounter+generationIncrement).ToString());
                policyNetwork.SaveNetwork(modelName+"-P" + (generationCounter+generationIncrement).ToString());

                valueInputs.Clear();
                valueOutputs.Clear();
                policyInputs.Clear();
                policyOutputs.Clear();
                uniquePositions.Clear();

                numRedWins = 0;
                numDraws = 0;
                numYellowWins = 0;
                
                epochCounter = 0;
                counter = 0;
                startTime = Time.realtimeSinceStartup;

                for (int i = 0; i < 42; i++) {
                    numPositions[i] = 0;
                }
            }
        }
    }

    public static float[] GetPosition(ulong redBitboard, ulong yellowBitboard)
    {
        // Create the output array for 85 floats (42 for red, 42 for yellow, 1 for parity)
        float[] result = new float[85];

        int totalPieces = 0;
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

                if (isRed || isYellow) totalPieces++;
            }
        }

        // Add parity as the 85th input (1.0 for red to play, 0.0 for yellow to play)
        result[84] = (totalPieces % 2 == 0) ? 1.0f : 0.0f;

        return result;
    }

    public float[] GetSymmetricalPosition(ulong redBitboard, ulong yellowBitboard)
    {
        // First get the neural network input representation
        float[] nnInput = GetPosition(redBitboard, yellowBitboard);
        float[] symmetricInput = new float[85];

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

        // Copy the parity bit (doesn't change with symmetry)
        symmetricInput[84] = nnInput[84];

        return symmetricInput;
    }

    public static void ShowBoardFromNNInput(float[] nnInput)
    {
        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        if (nnInput[84] == 1) {
            sb.AppendLine("Board State from NN Input (R):");
        } else {
            sb.AppendLine("Board State from NN Input (Y):");
        }
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