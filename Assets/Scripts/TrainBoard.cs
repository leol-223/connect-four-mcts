using UnityEngine;
using System.Collections.Generic;
using System;
using System.Threading.Tasks;
using System.Threading;
using System.Linq;
using UnityEngine.Profiling;
using System.Collections;

public class TrainBoard : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    public NeuralNetwork valueNetwork;
    public NeuralNetwork policyNetwork;
    public int numGenerations;
    public int startGeneration;
    public int generationIncrement;
    public int numGames;
    public int maxEpochs;
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

    public TMPro.TMP_Text sideDisplayMainText;
    public TMPro.TMP_Text sideDisplaySecondaryText;

    public GameObject maskPrefab;
    public GameObject tokenPrefab;
    public GameObject boardPrefab;
    public GameObject pointerPrefab;

    public Color redColor;
    public Color yellowColor;
    public Color redPointerColor;
    public Color yellowPointerColor;

    private GameObject pointer;
    private GameObject pointer2;
    private List<GameObject> tokens;
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
    private GameObject valueGraph;
    private GameObject policyGraph;
    private List<float> valueCosts;
    private List<float> policyCosts;
    private BoardNN board;

    private int epochCounter = 0;
    private int counter = 0;
    private int generationCounter = 0;
    private float startTime;

    private int numRedWins;
    private int numYellowWins;
    private int numDraws;

    private float[] lastRandomPosition;
    private float lastOutcome;
    
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
    
    // Add this field at the class level
    private bool isCleaningGeneration = false;

    void Start()
    {
        lastOutcome = 0f;
        lastRandomPosition = new float[85];
        TreeNode.SetExplorationConstraints(false);

        numPositions = new int[42];
        for (int i = 0; i < 42; i++) {
            numPositions[i] = 0;
        }

        int[] valueLayers = new int[] { 85, 64, 32, 16, 1};
        valueNetwork = new NeuralNetwork(
            valueLayers,
            NeuralNetwork.ActivationType.ReLU,
            NeuralNetwork.ActivationType.Tanh,
            NeuralNetwork.LossType.MSE
        );


        int[] policyLayers = new int[] { 85, 64, 32, 16, 7};
        policyNetwork = new NeuralNetwork(
            policyLayers,
            NeuralNetwork.ActivationType.ReLU,
            NeuralNetwork.ActivationType.Softmax,
            NeuralNetwork.LossType.MSE
        );

        if (startGeneration+generationIncrement > 0) {
            valueNetwork = NeuralNetwork.Load(modelName+"-V" + (startGeneration+generationIncrement).ToString());
            policyNetwork = NeuralNetwork.Load(modelName+"-P" + (startGeneration+generationIncrement).ToString());
            generationCounter = startGeneration;
            TreeNode.SetUniformEvaluation(false);
        } else {
            TreeNode.SetUniformEvaluation(true);
        }

        valueInputs = new List<float[]>();
        valueOutputs = new List<float[]>();

        policyInputs = new List<float[]>();
        policyOutputs = new List<float[]>();

        board = new BoardNN();

        startTime = Time.realtimeSinceStartup;
        uniquePositions = new HashSet<BoardState>();
        
        numRedWins = 0;
        numYellowWins = 0;
        numDraws = 0;

        // Right shift to fit game display view on left
        RectTransform gamesRect = gamesDisplay.GetComponent<RectTransform>();
        // Add 5,000 cause text width is 10,000
        gamesRect.anchoredPosition = new Vector2(150f+5000f, gamesRect.anchoredPosition.y);
        gamesDisplay.alignment = TMPro.TextAlignmentOptions.Left;
        
        RectTransform secondaryRect = secondaryDisplay.GetComponent<RectTransform>();
        secondaryRect.anchoredPosition = new Vector2(150f+5000f, secondaryRect.anchoredPosition.y);
        secondaryDisplay.alignment = TMPro.TextAlignmentOptions.Left;

        RectTransform generationRect = generationText.GetComponent<RectTransform>();
        // Add 5,000 cause text width is 10,000
        generationRect.anchoredPosition = new Vector2(150f+5000f, generationRect.anchoredPosition.y);
        generationText.alignment = TMPro.TextAlignmentOptions.Left;
        
        RectTransform thirdRect = thirdText.GetComponent<RectTransform>();
        thirdRect.anchoredPosition = new Vector2(150f+5000f, thirdRect.anchoredPosition.y);
        thirdText.alignment = TMPro.TextAlignmentOptions.Left;

        RectTransform statsRect = statsText.GetComponent<RectTransform>();
        // Add 5,000 cause text width is 10,000
        statsRect.anchoredPosition = new Vector2(150f+5000f, statsRect.anchoredPosition.y);
        statsText.alignment = TMPro.TextAlignmentOptions.Left;

        RectTransform sideDisplayRect = sideDisplayMainText.GetComponent<RectTransform>();
        Vector3 worldPosition = new Vector3(-3.3f, -2.2f, 0f); // Use the same x-coordinate as your game object
        Vector3 screenPosition = Camera.main.WorldToScreenPoint(worldPosition);
        sideDisplayRect.position = screenPosition;

        RectTransform sideDisplayRect2 = sideDisplaySecondaryText.GetComponent<RectTransform>();
        Vector3 worldPosition2causeunityisalittlebitchandwontletmeusethesamevariablenametwice = new Vector3(-3.3f, -2.8f, 0f); // Use the same x-coordinate as your game object
        Vector3 screenPosition2 = Camera.main.WorldToScreenPoint(worldPosition2causeunityisalittlebitchandwontletmeusethesamevariablenametwice);
        sideDisplayRect2.position = screenPosition2;

        tokens = new List<GameObject>();
        CreateEmptyBoard();

        TreeNode.ClearTranspositionTable();
        valueCosts = new List<float>();
        policyCosts = new List<float>();
    }

    // Update is called once per frame
    void Update()
    {
        if (generationCounter < numGenerations)
        {
            if (isCleaningGeneration)
            {
                return; // Skip update while cleaning
            }

            generationText.text = "Generation " + (generationCounter + generationIncrement + 1).ToString();
            if (counter < numGames)
            {
                int currentIterations = Mathf.Min(
                    maxIterations,
                    Mathf.RoundToInt((float)(initialIterations * Mathf.Pow(iterationsGrowthRate, generationCounter)))
                );

                float temperature = Mathf.Max(
                    minTemperature,
                    initialTemperature * Mathf.Pow(temperatureDecay, generationCounter)
                );


                secondaryDisplay.text = "Epoch 0: -";

                float currentNoise = initialRootNoise * Mathf.Pow(rootNoiseDecay, generationCounter);
                float currentAlpha = initialDirichletAlpha * Mathf.Pow(dirichletAlphaDecay, generationCounter);
                
                board.ResetBoard();
                board.valueNetwork = valueNetwork;
                board.policyNetwork = policyNetwork;
                board.rootNoise = currentNoise;
                board.dirichletAlpha = currentAlpha;

                List<float[]> positions = new List<float[]>();
                List<float[]> promises = new List<float[]>();
                BoardNN.Player winningPlayer;
                BoardNN.Player currentPlayer = BoardNN.Player.Red;
                bool debuggedOutput = false;

                int positionCount = 0;

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

                    HashSet<int> fullChildren = new HashSet<int>();

                    for (int i = 0; i < 7; i++) {
                        if (board.heights[i] == 6) {
                            fullChildren.Add(i);
                            break;
                        }
                    }

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
                    
                    float[] promise = new float[7];
                    for (int i = 0; i < 7; i++) {
                        promise[i] = board.promise[i];
                    }
                    positions.Add(position);

                    promises.Add(promise);
                    policyInputs.Add(position);
                    policyOutputs.Add(promise);
                    
                    if (position != symmetricalPosition) {
                        policyInputs.Add(symmetricalPosition);
                        policyOutputs.Add(promise.Reverse().ToArray());
                        positions.Add(symmetricalPosition);
                        promises.Add(promise.Reverse().ToArray());
                    }

                    positionCount += 1;
                }

                numPositions[positionCount - 1] += 1;

                float[] output;
                int moveCount = positionCount; // This represents how many moves it took
                float discount = Mathf.Pow(0.99f, moveCount);
                
                if (winningPlayer == BoardNN.Player.Red)
                {
                    output = new float[1] { 1.0f * discount };
                    numRedWins += 1;
                }
                else if (winningPlayer == BoardNN.Player.Yellow)
                {
                    output = new float[1] { -1.0f * discount };
                    numYellowWins += 1;
                }
                else
                {
                    output = new float[1] {0.0f};
                    numDraws += 1;
                }
                lastOutcome = output[0]+0f;

                int randomPositionIndex;
                randomPositionIndex = UnityEngine.Random.Range(0, positions.Count-1);
                float[] randomPosition = positions[randomPositionIndex];

                lastRandomPosition = (float[])randomPosition.Clone();
                DisplayPosition(randomPosition);
                
                // Count total pieces to determine whose turn it is
                int totalPieces = 0;
                for (int i = 0; i < 84; i++) {
                    if (randomPosition[i] > 0.5f) totalPieces++;
                }
                
                float[] policy = (float[])policyNetwork.Forward(randomPosition);
                float[] truePolicy = promises[randomPositionIndex];

                int maxIndex2 = 0;
                float maxValue2 = 0;
                int maxIndex3 = 0;
                float maxValue3 = 0;
                for (int i = 0; i < 7; i++) {
                    if (policy[i] > maxValue2) {
                        maxValue2 = policy[i];
                        maxIndex2 = i;
                    }
                    if (truePolicy[i] > maxValue3) {
                        maxValue3 = truePolicy[i];
                        maxIndex3 = i;
                    }
                }
                if (totalPieces % 2 == 0) {
                    sideDisplayMainText.color = redColor;
                    sideDisplayMainText.text = "Red to play";
                    
                    DisplayPointer(BoardNN.Player.Red, maxIndex2, 0.8f);
                    DisplayPointer2(BoardNN.Player.Red, maxIndex3, 0.5f);
                    
                } else {
                    // other yellow is too bright against white
                    sideDisplayMainText.color = new Color(182f/255f, 161f/255f, 27f/255f);
                    sideDisplayMainText.text = "Yellow to play";
                    
                    DisplayPointer(BoardNN.Player.Yellow, maxIndex2, 0.8f);
                    DisplayPointer2(BoardNN.Player.Yellow, maxIndex3, 0.5f);
                    
                }
                float evaluation = ((float[])valueNetwork.Forward(randomPosition))[0];
                
                string evalSign = evaluation > 0 ? "+" : "";
                string outSign = output[0] > 0 ? "+" : "";
                sideDisplaySecondaryText.text = $"Eval: {evalSign}{evaluation:F4}\nOutcome: {outSign}{output[0]:F4}";

                for (int i = 0; i < positions.Count; i++)
                {
                    valueInputs.Add(positions[i]);
                    valueOutputs.Add(output);
                }


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

                // Clear temporary game state lists
                positions.Clear();
                promises.Clear();
                
                // Force garbage collection periodically
                if (counter % 100 == 0)
                {
                    positions.Clear();
                    promises.Clear();
                    System.GC.Collect();
                    System.GC.WaitForPendingFinalizers();
                    Resources.UnloadUnusedAssets();

                    // LogMemorySnapshot($"{counter}");
                }
            }
            else if (epochCounter < maxEpochs)
            {
                thirdText.text = "";

                int trainTestSplit = (int)(trainRatio * valueInputs.Count);

                // Base learning rate decays with generations
                float generationAdjustedLR = initialLearningRate * Mathf.Pow(generationLearningRateDecay, generationCounter);
                // Further decay within epochs
                float learningRate = generationAdjustedLR * Mathf.Pow(epochLearningRateDecay, epochCounter);

                valueTrainInputs = valueInputs.GetRange(0, trainTestSplit);
                valueTestInputs = valueInputs.GetRange(trainTestSplit, valueInputs.Count - trainTestSplit);
                valueTrainOutputs = valueOutputs.GetRange(0, trainTestSplit);
                valueTestOutputs = valueOutputs.GetRange(trainTestSplit, valueOutputs.Count - trainTestSplit);

                float[][] valueTrainInputsArray = valueTrainInputs.ToArray();
                float[][] valueTestInputsArray = valueTestInputs.ToArray();
                float[][] valueTrainOutputsArray = valueTrainOutputs.ToArray();
                float[][] valueTestOutputsArray = valueTestOutputs.ToArray();
                // Debug.Log($"Value - Training data length: {valueTrainInputsArray.Length}, Testing data length: {valueTestInputsArray.Length}");

                valueNetwork.TrainEpoch(valueTrainInputsArray, valueTrainOutputsArray, batchSize: 64, learningRate: learningRate);

                float evaluation = ((float[])valueNetwork.Forward(lastRandomPosition))[0];

                string evalSign = evaluation > 0 ? "+" : "";
                string outSign = lastOutcome > 0 ? "+" : "";
                sideDisplaySecondaryText.text = $"Eval: {evalSign}{evaluation:F4}\nOutcome: {outSign}{lastOutcome:F4}";

                // Calculate validation loss
                float cost = 0;

                for (int j = 0; j < valueTestInputs.Count; j++) {
                    float[] valueOutput = (float[])valueNetwork.Forward(valueTestInputs[j]);
                    cost += CalculateLoss(valueOutput, valueTestOutputs[j]);
                }
                cost /= valueTestInputs.Count;

                valueCosts.Add(cost);

                int minCostIndex = 0;
                float minCost = Mathf.Infinity;
                for (int i = 0; i < valueCosts.Count; i++) {
                    if (valueCosts[i] < minCost) {
                        minCost = valueCosts[i];
                        minCostIndex = i;
                    }
                }
                
                Debug.Log("Value cost: " + cost.ToString());

                if (minCostIndex == valueCosts.Count - 1) {
                    valueNetwork.Save(modelName+"-V" + (generationCounter+generationIncrement+1).ToString());
                    Debug.Log($"Saving value network at epoch {epochCounter+1}");
                }

                if (valueCosts.Count > 1) {
                    if (valueGraph != null) {
                        Destroy(valueGraph);
                    }
                    valueGraph = CreateGraph(new Vector3(3.65f, -2.1f, 0), new Vector3(5.65f, -1.5f, 0), valueCosts, new Color(3f/255f, 102f/255f, 252f/255f), 0.08f);
                }

                policyTrainInputs = policyInputs.GetRange(0, trainTestSplit);
                policyTestInputs = policyInputs.GetRange(trainTestSplit, policyInputs.Count - trainTestSplit);
                policyTrainOutputs = policyOutputs.GetRange(0, trainTestSplit);
                policyTestOutputs = policyOutputs.GetRange(trainTestSplit, policyOutputs.Count - trainTestSplit);

                // Convert Lists to arrays
                float[][] policyTrainInputsArray = policyTrainInputs.ToArray();
                float[][] policyTestInputsArray = policyTestInputs.ToArray();
                float[][] policyTrainOutputsArray = policyTrainOutputs.ToArray();
                float[][] policyTestOutputsArray = policyTestOutputs.ToArray();

                // Debug.Log($"Policy - Training data length: {policyTrainInputsArray.Length}, Testing data length: {policyTrainOutputsArray.Length}");
                

                policyNetwork.TrainEpoch(policyTrainInputsArray, policyTrainOutputsArray, batchSize: 32, learningRate: learningRate);

                // Calculate validation loss
                cost = 0;

                for (int j = 0; j < policyTestInputs.Count; j++) {
                    float[] policyOutput = (float[])policyNetwork.Forward(policyTestInputs[j]);
                    cost += CalculateCrossEntropyLoss(policyOutput, policyTestOutputs[j]);
                }

                cost /= policyTestInputs.Count;

                policyCosts.Add(cost);

                minCostIndex = 0;
                minCost = Mathf.Infinity;
                for (int i = 0; i < policyCosts.Count; i++) {
                    if (policyCosts[i] < minCost) {
                        minCost = policyCosts[i];
                        minCostIndex = i;
                    }
                }

                Debug.Log("Policy cost: " + cost.ToString());
                if (minCostIndex == policyCosts.Count - 1) {
                    policyNetwork.Save(modelName+"-P" + (generationCounter+generationIncrement+1).ToString());
                    Debug.Log($"Saving policy network at epoch {epochCounter+1}");
                }

                if (policyCosts.Count > 1) {
                    if (policyGraph != null) {
                        Destroy(policyGraph);
                    }
                    policyGraph = CreateGraph(new Vector3(5.85f, -2.1f, 0), new Vector3(7.85f, -1.5f, 0), policyCosts, new Color(256f/255f, 61f/255f, 61f/255f), 0.08f);
                }

                epochCounter += 1;

                secondaryDisplay.text = "Epoch " + epochCounter.ToString() + " | V: " + valueCosts[valueCosts.Count-1].ToString("F4") + " | P: " + policyCosts[policyCosts.Count-1].ToString("F4");
            }
            else
            {
                StartCoroutine(CleanupGeneration());
            }
        }
    }

    private IEnumerator CleanupGeneration()
    {
        isCleaningGeneration = true;
        
        generationCounter += 1;

        // First, clear all current game objects and references
        if (board != null)
        {
            board.Dispose();
            board = null;
        }

        // Clear all collections explicitly
        valueInputs?.Clear();
        valueOutputs?.Clear();
        policyInputs?.Clear();
        policyOutputs?.Clear();
        valueTrainInputs?.Clear();
        valueTrainOutputs?.Clear();
        valueTestInputs?.Clear();
        valueTestOutputs?.Clear();
        policyTrainInputs?.Clear();
        policyTrainOutputs?.Clear();
        policyTestInputs?.Clear();
        policyTestOutputs?.Clear();

        // Null out all collections after clearing
        valueInputs = null;
        valueOutputs = null;
        policyInputs = null;
        policyOutputs = null;
        valueTrainInputs = null;
        valueTrainOutputs = null;
        valueTestInputs = null;
        valueTestOutputs = null;
        policyTrainInputs = null;
        policyTrainOutputs = null;
        policyTestInputs = null;
        policyTestOutputs = null;

        // Clear and null Unity objects
        if (valueGraph != null) {
            Destroy(valueGraph);
            valueGraph = null;
        }
        if (policyGraph != null) {
            Destroy(policyGraph);
            policyGraph = null;
        }
        
        // Clear all tokens
        foreach (var token in tokens) {
            if (token != null) {
                Destroy(token);
            }
        }
        tokens.Clear();
        tokens = null;

        // Clear neural networks
        valueNetwork = null;
        policyNetwork = null;

        // Clear static caches
        TreeNode.ClearTranspositionTable();
        TreeNode.SetUniformEvaluation(false);

        // First pass GC
        GC.Collect(2, GCCollectionMode.Forced, true);
        GC.WaitForPendingFinalizers();
        yield return Resources.UnloadUnusedAssets();
        
        yield return new WaitForSeconds(0.1f); // Give Unity time to process
        
        // Second pass GC
        GC.Collect(2, GCCollectionMode.Forced, true);
        GC.WaitForPendingFinalizers();
        yield return Resources.UnloadUnusedAssets();

        yield return new WaitForSeconds(0.1f); // Give Unity more time

        // Recreate all collections
        valueInputs = new List<float[]>();
        valueOutputs = new List<float[]>();
        policyInputs = new List<float[]>();
        policyOutputs = new List<float[]>();
        valueTrainInputs = new List<float[]>();
        valueTrainOutputs = new List<float[]>();
        valueTestInputs = new List<float[]>();
        valueTestOutputs = new List<float[]>();
        policyTrainInputs = new List<float[]>();
        policyTrainOutputs = new List<float[]>();
        policyTestInputs = new List<float[]>();
        policyTestOutputs = new List<float[]>();
        uniquePositions = new HashSet<BoardState>();
        tokens = new List<GameObject>();
        valueCosts = new List<float>();
        policyCosts = new List<float>();

        // Reset counters
        numRedWins = 0;
        numDraws = 0;
        numYellowWins = 0;
        epochCounter = 0;
        counter = 0;
        startTime = Time.realtimeSinceStartup;

        // Reset arrays
        numPositions = new int[42];
        lastRandomPosition = new float[85];

        // Third pass GC after recreation
        GC.Collect(2, GCCollectionMode.Forced, true);
        GC.WaitForPendingFinalizers();
        yield return Resources.UnloadUnusedAssets();

        // Load fresh networks
        valueNetwork = NeuralNetwork.Load(modelName+"-V" + (generationCounter+generationIncrement).ToString());
        policyNetwork = NeuralNetwork.Load(modelName+"-P" + (generationCounter+generationIncrement).ToString());

        isCleaningGeneration = false;

        board = new BoardNN();
    }

    public static float[] GetPosition(ulong redBitboard, ulong yellowBitboard)
    {
        float[] result = new float[2 * 6 * 7 + 1];  // 85 elements total (84 board state + 1 parity)

        int totalPieces = 0;
        for (int row = 0; row < 6; row++)
        {
            for (int col = 0; col < 7; col++)
            {
                int bitPosition = 8 * col + row;
                
                bool isRed = (redBitboard & ((ulong)1 << bitPosition)) != 0;
                bool isYellow = (yellowBitboard & ((ulong)1 << bitPosition)) != 0;
                
                // First 42 elements (0-41) represent red pieces
                // Last 42 elements (42-83) represent yellow pieces
                result[row * 7 + col] = isRed ? 1.0f : 0.0f;                  // Red channel
                result[42 + row * 7 + col] = isYellow ? 1.0f : 0.0f;         // Yellow channel
                
                if (isRed || isYellow) totalPieces++;
            }
        }
        
        // Add parity bit as 85th element (1.0 for red's turn, 0.0 for yellow's turn)
        result[84] = (totalPieces % 2 == 0) ? 1.0f : 0.0f;

        return result;
    }

    public static float[] GetSymmetricalPosition(ulong redBitboard, ulong yellowBitboard)
    {
        float[] result = new float[2 * 6 * 7 + 1];  // 85 elements total (84 board state + 1 parity)

        int totalPieces = 0;
        for (int row = 0; row < 6; row++)
        {
            for (int col = 0; col < 7; col++)
            {
                int bitPosition = 8 * col + row;
                
                bool isRed = (redBitboard & ((ulong)1 << bitPosition)) != 0;
                bool isYellow = (yellowBitboard & ((ulong)1 << bitPosition)) != 0;
                
                // Mirror horizontally by using (6 - col) instead of col
                result[row * 7 + (6 - col)] = isRed ? 1.0f : 0.0f;                  // Red channel
                result[42 + row * 7 + (6 - col)] = isYellow ? 1.0f : 0.0f;         // Yellow channel
                
                if (isRed || isYellow) totalPieces++;
            }
        }
        
        // Add parity bit as 85th element (1.0 for red's turn, 0.0 for yellow's turn)
        result[84] = (totalPieces % 2 == 0) ? 1.0f : 0.0f;

        return result;
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

    void CreateEmptyBoard() {
        float boardPadding = 0.1f;
        float boardScale = 0.75f;
        float xOffset = -3.3f;
        float yOffset = 0.7f;

        GameObject boardObject = Instantiate(boardPrefab);
        boardObject.transform.position = new Vector2(xOffset, yOffset);
        boardObject.transform.localScale = new Vector2(7 * boardScale + boardPadding * boardScale, 6 * boardScale + boardPadding * boardScale);

        GameObject collider = Instantiate(boardPrefab);
        collider.transform.position = new Vector2(xOffset, yOffset -(6 * boardScale) / 2f - 0.05f * boardScale);
        collider.transform.localScale = new Vector2(7 * boardScale + boardPadding * boardScale, 0.1f * boardScale);
        collider.AddComponent<BoxCollider2D>();
        // boardScale = size of one cell
        for (int i = 0; i < 7; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                GameObject maskObject = Instantiate(maskPrefab);
                maskObject.transform.position = new Vector2((i - 3f) * boardScale + xOffset, (j - 2.5f) * boardScale + yOffset);
                float scale = Mathf.Lerp(boardScale, 0, boardPadding);
                maskObject.transform.localScale = new Vector2(scale, scale);
            }
        }
    }

    void ClearBoard() {
        for (int i = 0; i < tokens.Count; i++) {
            Destroy(tokens[i]);
        }
        tokens.Clear();
    }

    void DrawToken(int x, int y, bool isRed) {
        GameObject token = Instantiate(tokenPrefab);

        float boardScale = 0.75f;
        float boardPadding = 0.1f;
        float xOffset = -3.3f;
        float yOffset = 0.7f;

        token.transform.position = new Vector2((x - 3f) * boardScale + xOffset, (y - 2.5f) * boardScale + yOffset);
        float scale = Mathf.Lerp(boardScale, 0, boardPadding);
        token.transform.localScale = new Vector2(scale, scale);

        if (isRed)
        {
            token.GetComponent<SpriteRenderer>().color = redColor;
        }
        else
        {
            token.GetComponent<SpriteRenderer>().color = yellowColor;
        }

        tokens.Add(token);
    }

    void DisplayPosition(float[] nnInput) {
        ClearBoard();  // Clear existing tokens first
        
        // For each column and row
        for (int row = 0; row < 6; row++) {
            for (int col = 0; col < 7; col++) {
                // Calculate indices in the flattened array
                int redIndex = row * 7 + col;        // First 42 values are red
                int yellowIndex = 42 + row * 7 + col;  // Next 42 values are yellow
                
                bool isRed = nnInput[redIndex] > 0.5;
                bool isYellow = nnInput[yellowIndex] > 0.5;
                
                if (isRed) {
                    DrawToken(col, row, true);
                } else if (isYellow) {
                    DrawToken(col, row, false);
                }
            }
        }
    }

    public void DisplayPointer(BoardNN.Player player, int position, float alpha) {
        if (pointer == null)
        {
            pointer = Instantiate(pointerPrefab);
        }
        if (player == BoardNN.Player.Red)
        {
            pointer.GetComponent<SpriteRenderer>().color = new Color(redPointerColor.r, redPointerColor.g, redPointerColor.b, alpha);
        }
        else {
            pointer.GetComponent<SpriteRenderer>().color = new Color(yellowPointerColor.r, yellowPointerColor.g, yellowPointerColor.b, alpha);
        }
        float xPos = ColumnToX(position);
        pointer.transform.position = new Vector2(xPos, -1.85f);
    }

    public void DisplayPointer2(BoardNN.Player player, int position, float alpha) {
        if (pointer2 == null)
        {
            pointer2 = Instantiate(pointerPrefab);
        }
        if (player == BoardNN.Player.Red)
        {
            pointer2.GetComponent<SpriteRenderer>().color = new Color(redPointerColor.r, redPointerColor.g, redPointerColor.b, alpha);
        }
        else {
            pointer2.GetComponent<SpriteRenderer>().color = new Color(yellowPointerColor.r, yellowPointerColor.g, yellowPointerColor.b, alpha);
        }
        float xPos = ColumnToX(position);
        pointer2.transform.position = new Vector2(xPos, -1.85f);
    }

    float ColumnToX(int column) {
        return (column - 3f) * 0.75f - 3.3f;
    }

    // Helper method
    private string GetNonZeroPositions(float[,,] input)
    {
        var positions = new List<string>();
        for (int c = 0; c < input.GetLength(0); c++)
            for (int i = 0; i < input.GetLength(1); i++)
                for (int j = 0; j < input.GetLength(2); j++)
                    if (Mathf.Abs(input[c,i,j]) > 1e-6)
                        positions.Add($"({c},{i},{j})={input[c,i,j]:F2}");
        return string.Join("; ", positions);
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

    void ClearAllCaches() {
        // Clear static caches
        TreeNode.ClearTranspositionTable();
        
        // Clear other collections
        valueInputs.Clear();
        valueOutputs.Clear();
        policyInputs.Clear();
        policyOutputs.Clear();
        
        // Force cleanup
        System.GC.Collect(2, GCCollectionMode.Forced, true);
        System.GC.WaitForPendingFinalizers();
        Resources.UnloadUnusedAssets();
    }

    void OnDestroy() {
        ClearAllCaches();
    }

    public GameObject CreateGraph(Vector3 bottomLeft, Vector3 topRight, List<float> heights, Color color, float width = 0.1f) {
        float minHeight = Mathf.Infinity;
        float maxHeight = Mathf.NegativeInfinity;
        for (int i = 0; i < heights.Count; i++) {
            float height = heights[i];
            minHeight = Mathf.Min(height, minHeight);
            maxHeight = Mathf.Max(height, maxHeight);
        }
        Vector3[] points = new Vector3[heights.Count];
        for (int i = 0; i < heights.Count; i++) {
            float x = Mathf.Lerp(bottomLeft.x, topRight.x, i /(float)(heights.Count));
            float yRelative = (heights[i] - minHeight) / (maxHeight - minHeight);
            float y = Mathf.Lerp(bottomLeft.y, topRight.y, yRelative);
            points[i] = new Vector3(x, y, 0);
        }
        return CreatePath(points, color, width);
    }

    public GameObject CreatePath(Vector3[] points, Color color, float width = 0.1f)
    {
        GameObject lineObj = new GameObject("Path");
        LineRenderer line = lineObj.AddComponent<LineRenderer>();
        
        // Set material
        line.material = new Material(Shader.Find("Unlit/Color"));
        line.material.color = color;
        
        // Set width
        line.startWidth = width;
        line.endWidth = width;
        
        // Set positions
        line.positionCount = points.Length;
        line.SetPositions(points);
        
        return lineObj;
    }

    void LogMemoryUsage(string point) {
        float totalMemory = (float)System.GC.GetTotalMemory(false) / (1024 * 1024); // MB
        Debug.Log($"Memory at {point}: {totalMemory:F2}MB");
    }

    private void LogMemorySnapshot(string point)
    {
        float totalMemoryMB = (float)GC.GetTotalMemory(false) / (1024 * 1024);
        float monoMemoryMB = UnityEngine.Profiling.Profiler.GetMonoUsedSizeLong() / (1024 * 1024);
        float totalAllocatedMB = UnityEngine.Profiling.Profiler.GetTotalAllocatedMemoryLong() / (1024 * 1024);
        
        Debug.Log($"Memory at {point}:");
        Debug.Log($"Total GC Memory: {totalMemoryMB:F2}MB");
        Debug.Log($"Mono Memory: {monoMemoryMB:F2}MB");
        Debug.Log($"Total Allocated: {totalAllocatedMB:F2}MB");
        
        // Log cache sizes
        Debug.Log($"Policy Cache Size: {TreeNode.policyCache?.Count ?? 0}");
        Debug.Log($"Value Cache Size: {TreeNode.valueCache?.Count ?? 0}");
    }
}