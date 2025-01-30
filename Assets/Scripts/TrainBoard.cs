using UnityEngine;
using System.Collections.Generic;
using System;
using System.Threading.Tasks;
using System.Threading;
using System.Linq;
using UnityEngine.Profiling;

public class TrainBoard : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
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

        valueNetwork = new NeuralNetwork(NeuralNetwork.MSE, NeuralNetwork.MSEDerivative);

        // Value Network
        valueNetwork.AddConvolutionalLayer(
            inputDepth: 2,    // One channel each for red and yellow pieces
            inputHeight: 6,   // 6 rows
            inputWidth: 7,    // 7 columns
            numFilters: 16,
            filterSize: 3,
            stride: 1,
            usePadding: true,
            activation: Activation.LeakyReLU
        );
        
        valueNetwork.AddPoolingLayer(
            inputDepth: 16,
            inputHeight: 6,
            inputWidth: 7,
            poolSize: 2,
            stride: 2
        );

        // Add an intermediate dense layer
        valueNetwork.AddDenseLayer(
            inputSize: 16 * 3 * 3,
            outputSize: 64,     // Wider first dense layer
            activation: Activation.LeakyReLU
        );

        // Add residual connection by concatenating with previous layer output
        valueNetwork.AddDenseLayer(
            inputSize: 64,
            outputSize: 1,
            activation: Activation.Tanh
        );

        // Mirror for policy network
        policyNetwork = new NeuralNetwork(NeuralNetwork.CategoricalCrossEntropy, NeuralNetwork.CategoricalCrossEntropyDerivative);

        // Policy Network
        policyNetwork.AddConvolutionalLayer(
            inputDepth: 2,
            inputHeight: 6,
            inputWidth: 7,
            numFilters: 16,
            filterSize: 3,
            stride: 1,
            usePadding: true,
            activation: Activation.LeakyReLU
        );
        
        policyNetwork.AddPoolingLayer(
            inputDepth: 16,
            inputHeight: 6,
            inputWidth: 7,
            poolSize: 2,
            stride: 2
        );

        policyNetwork.AddDenseLayer(
            inputSize: 16 * 3 * 3,
            outputSize: 64,
            activation: Activation.LeakyReLU
        );

        policyNetwork.AddDenseLayer(
            inputSize: 64,
            outputSize: 7,
            activation: Activation.Softmax
        );

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
                        if (fullChildren.Contains(i) && promise[i] > 0) {
                            Debug.Log($"Error encountered at child {i}, value {promise[i]}");
                            ShowBoardFromNNInput(position);
                        }
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
                    output = new float[1] { 1f * discount };
                    numRedWins += 1;
                }
                else if (winningPlayer == BoardNN.Player.Yellow)
                {
                    output = new float[1] { -1f * discount };
                    numYellowWins += 1;
                }
                else
                {
                    output = new float[1] {0f};
                    numDraws += 1;
                }

                if (debuggedOutput) {
                    Debug.Log(output[0]);
                }

                int randomPositionIndex;
                randomPositionIndex = UnityEngine.Random.Range(0, positions.Count-1);
                float[] randomPosition = positions[randomPositionIndex];
                DisplayPosition(randomPosition);
                // Count total pieces to determine whose turn it is
                int totalPieces = 0;
                for (int i = 0; i < 84; i++) {
                    if (randomPosition[i] > 0.5f) totalPieces++;
                }
                
                float[] policy = policyNetwork.Evaluate(randomPosition);
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
                float evaluation = valueNetwork.Evaluate(randomPosition)[0];
                
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

                valueNetwork.TrainOneEpoch(valueTrainInputs, valueTrainOutputs, currentLearningRate, 64);
                policyNetwork.TrainOneEpoch(policyTrainInputs, policyTrainOutputs, currentLearningRate, 64);

                float cost1 = valueNetwork.CalculateCost(valueTestInputs, valueTestOutputs);
                float cost2 = policyNetwork.CalculateCost(policyTestInputs, policyTestOutputs);
                
                epochCounter += 1;
                Debug.Log("Value cost: " + cost1.ToString());
                Debug.Log("Policy cost: " + cost2.ToString());
                secondaryDisplay.text = "Epoch " + epochCounter.ToString() + " | V: " + cost1.ToString("F4") + " | P: " + cost2.ToString("F4");
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
        float[] result = new float[84];

        for (int row = 0; row < 6; row++)
        {
            for (int col = 0; col < 7; col++)
            {
                int bitPosition = 8 * col + row;
                int nnPosition = row * 7 + col;  // Row-major ordering
                
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
                // Calculate source and target indices using row-major ordering
                int sourceIndex = row * 7 + col;
                int targetIndex = row * 7 + (6 - col);  // Flip column horizontally

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
        
        // For each row
        for (int row = 0; row < 6; row++) {
            // For each column
            for (int col = 0; col < 7; col++) {
                // Calculate position in the NN input array using row-major ordering
                int index = (row * 7) + col;
                
                // Check both red and yellow arrays
                bool isRed = nnInput[index] > 0.5f;
                bool isYellow = nnInput[index + 42] > 0.5f;
                
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
}