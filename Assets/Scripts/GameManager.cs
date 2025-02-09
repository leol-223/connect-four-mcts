using UnityEngine;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;

public class GameManager : MonoBehaviour
{
    public List<Board.Player> aiPlayers;
    public int searchTimeMilliseconds;
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
    public string valuePath1;
    public string policyPath1;
    public string valuePath2;
    public string policyPath2;
    private NeuralNetwork valueNetwork1;
    private NeuralNetwork policyNetwork1;
    private NeuralNetwork valueNetwork2;
    private NeuralNetwork policyNetwork2;
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
    private float searchTimeAtStart;
    private bool searchStarted = false;
    private float timeSinceLastSearchStart;
    private TreeNode rootNode;
    private Board.Player currentPlayer = Board.Player.Red;

    
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        ClearAllResources();
        
        TreeNode.ClearTranspositionTable();
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

        valueNetwork1 = NeuralNetwork.Load(valuePath1);
        policyNetwork1 = NeuralNetwork.Load(policyPath1);
        valueNetwork2 = NeuralNetwork.Load(valuePath2);
        policyNetwork2 = NeuralNetwork.Load(policyPath2);

        board.valueNetwork = valueNetwork1;
        board.policyNetwork = policyNetwork1;

        searchTimeAtStart = 0;
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
                        if (!searchStarted) {
                            if (aiPlayers.IndexOf(currentPlayer) == 0) {
                                board.valueNetwork = valueNetwork1;
                                board.policyNetwork = policyNetwork1;
                            } else {
                                board.valueNetwork = valueNetwork2;
                                board.policyNetwork = policyNetwork2;
                            }
                            rootNode = board.GetRootNode(currentPlayer == Board.Player.Red);
                            searchStarted = true;
                            searchTimeAtStart = Time.realtimeSinceStartup;
                            // only clear if there are two ai players
                            if (aiPlayers[0] != Board.Player.None && aiPlayers[1] != Board.Player.None) {
                                TreeNode.ClearTranspositionTable();
                            }
                        }

                        if (searchStarted)
                        {
                            timeSinceLastSearchStart = Time.realtimeSinceStartup;
                            try {
                                Parallel.For(0, 100, i => {
                                    rootNode.Search(0, 0);
                                });
                            }
                            catch (AggregateException ae) {
                                Debug.LogError($"Parallel search failed: {ae.Message}");
                            }

                            BoardNN.MoveEval move = board.BestMove(rootNode, temperature);

                            LogMemoryUsage();

                            DebugSearchStatistics(rootNode);
                            ShowPointer(currentPlayer, move.Move);
                            if (Time.realtimeSinceStartup - searchTimeAtStart >= searchTimeMilliseconds * 0.001f)
                            {
                                searchTimeAtStart = 0;
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
            if (winningPlayer != BoardNN.Player.None || board.IsFull())
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

    public void DebugSearchStatistics(TreeNode rootNode) {
        float totalVisits = rootNode.children.Sum(c => c.N);
        
        var stats = rootNode.children.Select(child => new {
            Column = child.priorMove + 1,
            Visits = child.N,
            Seldepth = child.GetMaxDepth(),
            Mindepth = child.GetMinDepth(),
            Value = child.Q,
            Prior = child.prior
        }).OrderByDescending(s => s.Visits);
        
        string statString = "";
        foreach (var stat in stats) {
            float normalizedValue = (stat.Value + 1f) / 2f;
            string valueColor = $"#{(int)(255):X2}{(int)(255 * (1-normalizedValue)):X2}00";
            
            statString += $"Col {stat.Column}: " +
                         $"Visits=<color=#7098DB>{stat.Visits}/{totalVisits+1}</color> | " +
                         $"Depth=<color=#B784D3>{stat.Mindepth}-{stat.Seldepth}</color> | " +
                         $"Value=<color={valueColor}>{stat.Value:F3}</color> | " +
                         $"Prior=<color=#75C7E5>{stat.Prior:F3}</color>\n";
        }
        Debug.Log(statString);
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

    void OnDestroy()
    {
        // Clear all resources
        TreeNode.ClearTranspositionTable();
        
        // Clear neural networks
        valueNetwork1 = null;
        policyNetwork1 = null;
        valueNetwork2 = null;
        policyNetwork2 = null;
        
        // Force garbage collection
        Resources.UnloadUnusedAssets();
        System.GC.Collect();
        System.GC.WaitForPendingFinalizers();
    }

    void OnDisable()
    {
        // Clear search state
        rootNode = null;
        board = null;
        TreeNode.ClearTranspositionTable();
    }

    void LogMemoryUsage() {
        long totalMemory = GC.GetTotalMemory(false);
        Debug.Log($"Total Memory: {totalMemory / 1024 / 1024}MB");
    }

    void ClearAllResources() {
        // Clear static caches
        TreeNode.ClearTranspositionTable();
        
        // Clear instance references
        rootNode = null;
        board = null;
        
        // Clear neural networks
        valueNetwork1 = null;
        policyNetwork1 = null;
        valueNetwork2 = null;
        policyNetwork2 = null;
        
        // Force multiple levels of garbage collection
        System.GC.Collect(2, GCCollectionMode.Forced, true);
        System.GC.WaitForPendingFinalizers();
        
        // Unity specific cleanup
        Resources.UnloadUnusedAssets();
    }
}
