using UnityEngine;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;

public class GameManager : MonoBehaviour
{
    public List<Board.Player> aiPlayers;
    public int searchIterations;
    public Color redColor;
    public Color yellowColor;
    public Color redPointerColor;
    public Color yellowPointerColor;
    public Color endScreenColor;
    public GameObject maskPrefab;
    public GameObject tokenPrefab;
    public GameObject boardPrefab;
    public GameObject squarePrefab;
    public GameObject pointerPrefab;
    public float boardScale;
    public float boardPadding;
    public float yLevel;
    public float dropHeight;
    public string valuePath;
    public string policyPath;
    public NeuralNetwork valueNetwork;
    public NeuralNetwork policyNetwork;
    public float temperature;
    public bool rootNoise;

    private float minTokenDelay = 0.1f;
    private BoardNN board;
    private GameObject pointer;
    private GameObject slidingToken;
    private GameObject endScreen;
    private GameObject[] tokenObjects;
    private float timeSinceEnd;
    private int currentIter = 0;
    private bool isEnded = false;
    private bool isFirstPlayer = true;
    private bool isSlidingStage = true;
    private float timeSinceTokenDrop;
    private bool searchStarted = false;
    private TreeNode rootNode;
    private Board.Player currentPlayer = Board.Player.Red;


    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        TreeNode.SetExplorationConstraints(false);

        if (!aiPlayers.Contains(Board.Player.Red))
        {
            slidingToken = Instantiate(tokenPrefab);
            slidingToken.transform.position = new Vector2(0f, dropHeight);
            slidingToken.GetComponent<SpriteRenderer>().color = redColor;
        }
        tokenObjects = new GameObject[42];
        CreateEmptyBoard();
        board = new BoardNN();
        rootNode = null;

        valueNetwork = new NeuralNetwork(NeuralNetwork.MSE, NeuralNetwork.MSEDerivative);

        // Value Network
        valueNetwork.AddConvolutionalLayer(
            inputDepth: 2,
            inputHeight: 6,
            inputWidth: 7,
            numFilters: 32,     // Increased from 16 to 32 filters
            filterSize: 3,
            stride: 1,
            usePadding: true,
            activation: Activation.LeakyReLU
        );
        
        valueNetwork.AddPoolingLayer(
            inputDepth: 32,
            inputHeight: 6,
            inputWidth: 7,
            poolSize: 2,
            stride: 2
        );

        valueNetwork.AddDenseLayer(
            inputSize: 32 * 3 * 3,
            outputSize: 64,     // Increased from 32 to 64
            activation: Activation.LeakyReLU
        );

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
            numFilters: 32,     // Increased to match value network
            filterSize: 3,
            stride: 1,
            usePadding: true,
            activation: Activation.LeakyReLU
        );
        
        policyNetwork.AddPoolingLayer(
            inputDepth: 32,
            inputHeight: 6,
            inputWidth: 7,
            poolSize: 2,
            stride: 2
        );

        policyNetwork.AddDenseLayer(
            inputSize: 32 * 3 * 3,
            outputSize: 64,     // Increased to match value network
            activation: Activation.LeakyReLU
        );

        policyNetwork.AddDenseLayer(
            inputSize: 64,
            outputSize: 7,
            activation: Activation.Softmax
        );

        valueNetwork.LoadNetwork(valuePath);
        policyNetwork.LoadNetwork(policyPath);

        board.valueNetwork = valueNetwork;
        board.policyNetwork = policyNetwork;
    }

    // Update is called once per frame
    void Update()
    {
        if (isSlidingStage && !aiPlayers.Contains(currentPlayer))
        {
            Vector2 pos = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            float leftBound = -(7 * boardScale) / 2f;
            float rightBound = (7 * boardScale) / 2f;
            float xPos = Mathf.Clamp(pos.x, leftBound + boardScale / 2f, rightBound - boardScale / 2f);
            slidingToken.transform.position = new Vector2(xPos, dropHeight);

            if (Input.GetKeyDown(KeyCode.Mouse0) && GetColumnProximity(xPos) < 0.25f && !isEnded)
            {
                int columnPosition = GetTokenColumn(xPos);
                MakeMove(columnPosition);
            }
        }
        else {
            timeSinceTokenDrop += Time.deltaTime;
            if (timeSinceTokenDrop >= minTokenDelay && !isEnded && !board.IsFull())
            {
                if (slidingToken == null || Mathf.Abs(slidingToken.GetComponent<Rigidbody2D>().linearVelocityY) < 0.0001f)
                {
                    if (aiPlayers.Contains(currentPlayer))
                    {
                        if (currentIter < searchIterations)
                        {
                            if (!searchStarted) {
                                rootNode = board.GetRootNode(currentPlayer == Board.Player.Red);
                                searchStarted = true;
                            }

                            try {
                                Parallel.For(0, 100, i => {
                                    rootNode.Search(0, 0);
                                });
                                currentIter += 100;
                            }
                            catch (AggregateException ae) {
                                Debug.LogError($"Parallel search failed: {ae.Message}");
                            }

                            BoardNN.MoveEval move = board.BestMove(rootNode, temperature);
                            Debug.Log(string.Join(",", board.promise));

                            DebugSearchStatistics(rootNode);
                            ShowPointer(currentPlayer, move.Move);
                            if (currentIter >= searchIterations)
                            {
                                currentIter = 0;
                                CreateToken();
                                MakeMove(move.Move);
                                Destroy(pointer);
                                searchStarted = false;
                            }
                        }
                    }
                    else
                    {
                        CreateToken();
                    }
                }
            }
        }

        if (isEnded)
        {
            timeSinceEnd += Time.deltaTime;
            // float oscillationTime = 1.5f;
            float oscillationTime = 180f / 130f;
            if (timeSinceEnd >= 1.2f && timeSinceEnd <= 1.2f + oscillationTime)// && timeSinceEnd <= 1.5f + oscillationTime * 3)
            {
                int[] winningPositions = GetConnectFourPositions();
                Vector3 rotation = new Vector3(0, 130f, 0) * Time.deltaTime;
                for (int i = 0; i < 4; i++) {
                    tokenObjects[winningPositions[i]].transform.Rotate(rotation);
                }
            }
            if (timeSinceEnd >= 1.7f + oscillationTime)
            {
                float currentAlpha = Mathf.Min((timeSinceEnd - 1.7f - oscillationTime) / 0.7f, endScreenColor.a);
                Color endScreenCurrent = new Color(endScreenColor.r, endScreenColor.g, endScreenColor.b, currentAlpha);
                endScreen.GetComponent<SpriteRenderer>().color = endScreenCurrent;
            }
        }
    }

    public void MakeMove(int position) {
        if (!board.IsValidMove(position) || board.GetWinningPlayer() != BoardNN.Player.None)
        {
            // Invalid move
        }
        else
        {
            if (isFirstPlayer)
            {
                board.MakeMove(position, BoardNN.Player.Red);
            }
            else
            {
                board.MakeMove(position, BoardNN.Player.Yellow);
            }
            BoardNN.Player winningPlayer = board.GetWinningPlayer();
            if (winningPlayer != BoardNN.Player.None)
            {
                isEnded = true;
                timeSinceEnd = 0;
                endScreen = Instantiate(squarePrefab);
                endScreen.transform.position = new Vector3(0, 0, -1);
                endScreen.transform.localScale = new Vector2(20, 10);
                Color endScreenInvis = new Color(endScreenColor.r, endScreenColor.g, endScreenColor.b, 0);
                endScreen.GetComponent<SpriteRenderer>().color = endScreenInvis;
                for (int i = 0; i < 42; i++)
                {
                    if (tokenObjects[i] != null) {
                        tokenObjects[i].GetComponent<Rigidbody2D>().constraints = RigidbodyConstraints2D.FreezePosition;
                    }
                }
            }
            float truePosition = ColumnToX(position);
            isSlidingStage = false;
            timeSinceTokenDrop = 0f;
            slidingToken.transform.position = new Vector2(truePosition, dropHeight);
            slidingToken.GetComponent<Rigidbody2D>().constraints = RigidbodyConstraints2D.FreezePositionX;
            tokenObjects[position * 6 + (board.heights[position] - 1)] = slidingToken;
            SwitchPlayer();
        }
        /*
        int[] drawIndexes = new int[] {0, 7, 14, 21};
        for (int i = 0; i < drawIndexes.Length; i++)
        {
            if (tokenObjects[drawIndexes[i]] != null) {
                tokenObjects[drawIndexes[i]].GetComponent<SpriteRenderer>().color = new Color(0, 0, 0);
            };
        }
        */
    }

    public void CreateToken() {
        slidingToken = Instantiate(tokenPrefab);
        slidingToken.transform.position = new Vector2(0f, dropHeight);
        if (isFirstPlayer)
        {
            slidingToken.GetComponent<SpriteRenderer>().color = redColor;
        }
        else
        {
            slidingToken.GetComponent<SpriteRenderer>().color = yellowColor;
        }
        isSlidingStage = true;
    }

    public void ShowPointer(Board.Player player, int position) {
        if (pointer == null)
        {
            pointer = Instantiate(pointerPrefab);
        }
        if (player == Board.Player.Red)
        {
            pointer.GetComponent<SpriteRenderer>().color = redPointerColor;
        }
        else {
            pointer.GetComponent<SpriteRenderer>().color = yellowPointerColor;
        }
        float xPos = ColumnToX(position);
        pointer.transform.position = new Vector2(xPos, pointer.transform.position.y);
    }

    public void SwitchPlayer() {
        isFirstPlayer = !isFirstPlayer;
        if (currentPlayer == Board.Player.Red)
        {
            currentPlayer = Board.Player.Yellow;
        }
        else {
            currentPlayer = Board.Player.Red;
        }
    }

    // Basically a copy of get winning moves from board but returning the positions
    // I didn't want to pollute my beautiful board class with this
    public int[] GetConnectFourPositions() {
        // Columns
        for (int i = 0; i < 7; i++)
        {
            for (int j = 0; j < board.heights[i] - 3; j++)
            {
                BoardNN.Player endingValue = board.GetBit(8 * i + j + 3);
                int[] currentChain = new int[4];
                currentChain[0] = 6 * i + j + 3;
                bool connectFour = true;
                for (int k = 0; k < 3; k++)
                {
                    if (board.GetBit(8 * i + j + k) != endingValue)
                    {
                        connectFour = false;
                        break;
                    }
                    currentChain[k + 1] = 6 * i + j + k;
                }
                if (connectFour)
                {
                    return currentChain;
                }
            }
        }
        // Rows
        for (int j = 0; j < 6; j++)
        {
            for (int i = 0; i < 4; i++)
            {
                BoardNN.Player endingValue = board.GetBit(8 * (i + 3) + j);
                int[] currentChain = new int[4];
                currentChain[0] = 6 * (i + 3) + j;
                if (endingValue == BoardNN.Player.None)
                {
                    break;
                }
                bool connectFour = true;
                for (int k = 0; k < 3; k++)
                {
                    if (board.GetBit(8 * (i + k) + j) != endingValue)
                    {
                        connectFour = false;
                        break;
                    }
                    currentChain[k + 1] = 6 * (i + k) + j;
                }
                if (connectFour)
                {
                    return currentChain;
                }
            }
        }
        // Diagonals going down-right
        for (int i = 0; i < 4; i++)
        {
            for (int j = 3; j < board.heights[i]; j++)
            {
                // Top left value
                BoardNN.Player endingValue = board.GetBit(8 * i + j);
                int[] currentChain = new int[4];
                currentChain[0] = 6 * i + j;
                bool connectFour = true;
                for (int k = 1; k < 4; k++)
                {
                    if (board.GetBit(8 * (i + k) + (j - k)) != endingValue)
                    {
                        connectFour = false;
                        break;
                    }
                    currentChain[k] = 6 * (i + k) + (j - k);
                }
                if (connectFour)
                {
                    return currentChain;
                }
            }
        }
        // Diagonals going down-left
        for (int i = 3; i < 7; i++)
        {
            for (int j = 3; j < board.heights[i]; j++)
            {
                // Top right value
                BoardNN.Player endingValue = board.GetBit(8 * i + j);
                int[] currentChain = new int[4];
                currentChain[0] = 6 * i + j;
                bool connectFour = true;
                for (int k = 1; k < 4; k++)
                {
                    if (board.GetBit(8 * (i - k) + (j - k)) != endingValue)
                    {
                        connectFour = false;
                        break;
                    }
                    currentChain[k] = 6 * (i - k) + (j - k);
                }
                if (connectFour)
                {
                    return currentChain;
                }
            }
        }
        return new int[4] { -1, -1, -1, -1 };
    }

    void CreateEmptyBoard() {
        GameObject boardObject = Instantiate(boardPrefab);
        boardObject.transform.position = new Vector2(0, yLevel);
        boardObject.transform.localScale = new Vector2(7 * boardScale + boardPadding * boardScale, 6 * boardScale + boardPadding * boardScale);

        GameObject collider = Instantiate(boardPrefab);
        collider.transform.position = new Vector2(0, yLevel - (6 * boardScale) / 2f - 0.05f * boardScale);
        collider.transform.localScale = new Vector2(7 * boardScale + boardPadding * boardScale, 0.1f * boardScale);
        collider.AddComponent<BoxCollider2D>();
        // boardScale = size of one cell
        for (int i = 0; i < 7; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                GameObject maskObject = Instantiate(maskPrefab);
                maskObject.transform.position = new Vector2((i - 3f) * boardScale, (j - 2.5f) * boardScale + yLevel);
                float scale = Mathf.Lerp(boardScale, 0, boardPadding);
                maskObject.transform.localScale = new Vector2(scale, scale);
            }
        }
    }

    int GetTokenColumn(float x) {
        return Mathf.FloorToInt((x + (7 * boardScale) / 2) / (boardScale));
    }

    float ColumnToX(int column) {
        return (column - 3f) * boardScale;
    }

    float GetColumnProximity(float x) {
        return Mathf.Abs(((x + (7 * boardScale) / 2) / boardScale) % 1 - 0.5f);
    }

    public float[] GetNeuralInput(ulong redBitboard, ulong yellowBitboard)
    {
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

    public void DebugPolicyOutputs(ulong redBitboard, ulong yellowBitboard)
    {
        float[] result = GetNeuralInput(redBitboard, yellowBitboard);

        float[] evaluation = policyNetwork.Evaluate(result);
        for (int i = 0; i < 7; i++)
        {
            Debug.Log("Column " + i + ": " + evaluation[i]);
        }
    }

    public void DebugSearchStatistics(TreeNode rootNode) {
        float totalVisits = rootNode.children.Sum(c => c.N);
        
        var stats = new List<string>();
        foreach (var child in rootNode.children) {
            int col = child.priorMove;
            float visits = child.N;
            float value = child.Q;
            float prior = child.prior;
            float visitPercentage = visits / totalVisits * 100;
            
            stats.Add($"Column {col + 1}: " +
                     $"Visits={visits} ({visitPercentage:F1}%), " +
                     $"Value={value:F3}, " +
                     $"Prior={prior:F3}");
        }
        
        // Sort by visit count for easier comparison
        stats.Sort((a, b) => {
            float visitsA = float.Parse(a.Split('=')[1].Split(' ')[0]);
            float visitsB = float.Parse(b.Split('=')[1].Split(' ')[0]);
            return visitsB.CompareTo(visitsA);
        });
        
        string statString = "";
        foreach (var stat in stats) {
            statString += (stat+"\n");
        }
        Debug.Log(statString);
    }
}