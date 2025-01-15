using UnityEngine;
using System.Collections.Generic;

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
    public int[] valueShape;
    public int[] policyShape;
    public NeuralNetwork valueNetwork;
    public NeuralNetwork policyNetwork;

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

        this.valueNetwork = new NeuralNetwork(valueShape, NeuralNetwork.ReLU, NeuralNetwork.Tanh, NeuralNetwork.ReLUDerivative, NeuralNetwork.TanhDerivative, NeuralNetwork.MSE, NeuralNetwork.MSEDerivative);
        this.policyNetwork = new NeuralNetwork(policyShape, NeuralNetwork.ReLU, NeuralNetwork.Softmax, NeuralNetwork.ReLUDerivative, NeuralNetwork.SoftmaxDerivative, NeuralNetwork.CategoricalCrossEntropy, NeuralNetwork.CategoricalCrossEntropyDerivative);

        this.valueNetwork.LoadNetwork(valuePath);
        this.policyNetwork.LoadNetwork(policyPath);

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
                                DebugPolicyOutputs(board.redBitboard, board.yellowBitboard);
                            }
                            /*
                            float startTime = Time.realtimeSinceStartup;
                            BoardNN.MoveEval move = board.TreeSearch(searchIterations, isRed);
                            float endTime = Time.realtimeSinceStartup;
                            */
                            for (int i = 0; i < 200; i++) {
                                rootNode.Search();
                                currentIter += 1;
                            }
                            BoardNN.MoveEval move = board.BestMove(rootNode);

                            Debug.Log($"Col {move.Move + 1} | Evaluation: {move.Eval} | Max depth: {board.maxDepth}");
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
        bool[] bits = new bool[128];

        for (int i = 0; i < 64; i++)
        {
            bits[i] = (redBitboard & (1UL << i)) != 0;
            bits[64 + i] = (yellowBitboard & (1UL << i)) != 0;
        }

        // Create the output array for 84 floats
        float[] result = new float[84];

        // Copy every 6 bits, skipping 2 buffer bits after every 6 bits
        int resultIndex = 0;
        for (int i = 0; i < bits.Length; i += 8)
        {
            for (int j = 0; j < 6; j++) // Copy 6 bits as floats
            {
                if (resultIndex < 84)
                {
                    result[resultIndex++] = bits[i + j] ? 1.0f : 0.0f;
                }
            }
            // Skip the 2 buffer bits
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
}
