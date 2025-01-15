using UnityEngine;
using System.Collections.Generic;
using System;

public delegate float EvaluationFunction(ulong redBitboard, ulong yellowBitboard);

public struct State
{
    public ulong redBitboard;
    public ulong yellowBitboard;
    public int[] heights;

    public State(ulong redBitboard, ulong yellowBitboard, int[] heights)
    {
        this.redBitboard = redBitboard;
        this.yellowBitboard = yellowBitboard;
        this.heights = heights;
    }
}

public class TreeNode {
    public List<TreeNode> children;
    public int priorMove;
    public State state;
    public float? eval;
    public float promise;
    public bool redToPlay;
    public bool isTerminal;
    public NeuralNetwork valueNetwork;
    public NeuralNetwork policyNetwork;
    public int visits;
    public int depth;
    public bool isRootNode;

    // MCTS-related:
    public float N;     // visit count
    public float W;     // total (accumulated) value
    public float Q;     // average value = W / N
    public float prior; // policy prior (from NN or heuristic)

    // For demonstration, we’ll define a constant for exploration:
    private const float C_PUCT = 1.4f;

    public TreeNode(NeuralNetwork valueNetwork, NeuralNetwork policyNetwork, State state) {
        this.valueNetwork = valueNetwork;
        this.policyNetwork = policyNetwork;
        this.state = state;
        visits = 0;
        isRootNode = false;
    }

    public void CreateChildren(float temperature) {
        List<int> possibleMoves = GetValidMoves();
        children = new List<TreeNode>();

        float[] policy = policyNetwork.Evaluate(StateToInput(state));

        for (int i = 0; i < possibleMoves.Count; i++)
        {
            int move = possibleMoves[i];
            State childState = MakeMove(move);
            TreeNode child = new TreeNode(valueNetwork, policyNetwork, childState);
            child.priorMove = move;
            child.redToPlay = !redToPlay;
            child.depth = depth + 1;

            // Check terminal states
            if (HasConnectFour(childState.redBitboard))
            {
                child.isTerminal = true;
                child.eval = child.redToPlay ? -(100f + child.depth * 0.01f) : 100f + child.depth * 0.01f;
            }
            else if (HasConnectFour(childState.yellowBitboard))
            {
                child.isTerminal = true;
                child.eval = child.redToPlay ? 100f + child.depth * 0.01f : -(100f + child.depth * 0.01f);
            }
            else if (IsFull(childState))
            {
                child.isTerminal = true;
                child.eval = 0f;
            }
            else
            {
                float[] input = StateToInput(childState);
                float nnValue = valueNetwork.Evaluate(input)[0];
                child.eval = child.redToPlay ? -nnValue : nnValue;
            }

            child.prior = policy[i];
            children.Add(child);
        }
    }


    public List<int> GetValidMoves()
    {
        List<int> moves = new List<int>();
        for (int i = 0; i < 7; i++)
        {
            if (state.heights[i] < 6)
            {
                moves.Add(i);
            }
        }
        return moves;
    }

    public State MakeMove(int column)
    {
        // First 6 = column 1 (row 1-6)
        // Next 6 = column 2 (row 1-6)
        // etc
        int[] newHeights = new int[7];
        Array.Copy(state.heights, newHeights, 7);

        State newState = new State(state.redBitboard, state.yellowBitboard, newHeights);
        int bitPosition = 8 * column + state.heights[column];
        if (redToPlay)
        {
            newState.redBitboard |= ((ulong)1 << bitPosition);
        }
        else
        {
            newState.yellowBitboard |= ((ulong)1 << bitPosition);
        }
        newState.heights[column] += 1;
        return newState;
    }

    public float[] StateToInput(State state) {
        bool[] bits = new bool[128];

        for (int i = 0; i < 64; i++)
        {
            bits[i] = (state.redBitboard & (1UL << i)) != 0;
            bits[64 + i] = (state.yellowBitboard & (1UL << i)) != 0;
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

    private bool HasConnectFour(ulong b)
    {
        // 1) Vertical check (shift by 1)
        {
            ulong m = b & (b >> 1);
            // If we can still find 2 more consecutive after that, there is a 4.
            if ((m & (m >> 2)) != 0UL)
                return true;
        }

        // 2) Horizontal check (shift by 6)
        {
            ulong m = b & (b >> 8);
            if ((m & (m >> 16)) != 0UL)
                return true;
        }

        // 3) Diagonal up-right (shift by 7)
        {
            ulong m = b & (b >> 9);
            if ((m & (m >> 18)) != 0UL)
                return true;
        }

        // 4) Diagonal up-left (shift by 5)
        {
            ulong m = b & (b >> 7);
            if ((m & (m >> 14)) != 0UL)
                return true;
        }

        return false;
    }

    public bool IsFull(State state)
    {
        for (int i = 0; i < 7; i++)
        {
            if (state.heights[i] < 6) {
                return false;
            }
        }
        return true;
    }

    public int GetMaxDepth() {
        if (children == null)
        {
            return depth;
        }
        int maxDepth = 0;
        for (int i = 0; i < children.Count; i++)
        {
            maxDepth = Mathf.Max(maxDepth, children[i].GetMaxDepth());
        }
        return maxDepth;
    }

    public void CreateChildrenWithRootNoise(float alpha, float epsilon)
    {
        // Normal child creation logic:
        CreateChildren(1.0f); // sets child.prior in some default way
                          // e.g. from NN policy or uniform

        // Now blend in Dirichlet noise at the root:
        float[] noise = NoiseUtils.SampleDirichlet(children.Count, alpha);

        for (int i = 0; i < children.Count; i++)
        {
            float oldPrior = children[i].prior;  // e.g. from NN
            float newPrior = (1 - epsilon) * oldPrior + epsilon * noise[i];
            children[i].prior = newPrior;
        }
    }

    public float Search()
    {
        if (isTerminal)
        {
            // For terminal nodes, we should still count the visit and update Q
            N += 1;
            W += (float)eval;
            Q = W / N;
            return (float)eval;
        }

        if (children == null)
        {
            if (isRootNode)
            {
                CreateChildrenWithRootNoise(0.3f, 0.25f);
            }
            else
            {
                CreateChildren(1.0f);
            }

            // For newly expanded nodes, check if any children are terminal
            foreach (var child in children)
            {
                if (child.isTerminal)
                {
                    // Initialize terminal nodes with one visit
                    child.N = 1;
                    child.W = (float)child.eval;
                    child.Q = child.W / child.N;
                }
            }

            float value = 0f;
            if (!isTerminal) {
                float[] input = StateToInput(state);
                value = valueNetwork.Evaluate(input)[0];
                value = redToPlay ? value : -value;
            }
            W += value;
            N += 1;
            Q = W / N;
            return value;
        }

        float totalVisits = 0;
        foreach (var c in children)
        {
            totalVisits += c.N;
        }

        TreeNode bestChild = null;
        float bestScore = float.NegativeInfinity;

        foreach (var c in children)
        {
            float uct;
            if (c.N == 0)
            {
                // For unvisited nodes, use their terminal value if available
                uct = c.isTerminal ? ((float)c.eval) : (C_PUCT * c.prior * Mathf.Sqrt(totalVisits + 1));
            }
            else
            {
                uct = c.Q + C_PUCT * c.prior * Mathf.Sqrt(totalVisits + 1) / (1 + c.N);
            }
            
            if (uct > bestScore)
            {
                bestScore = uct;
                bestChild = c;
            }
        }

        float childValue = -bestChild.Search();

        W += childValue;
        N += 1;
        Q = W / N;

        return childValue;
    }

}

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
    public int iterations;
    public string modelName;

    public TMPro.TMP_Text gamesDisplay;
    public TMPro.TMP_Text secondaryDisplay;
    public TMPro.TMP_Text generationText;

    private List<float[]> valueInputs;
    private List<float[]> valueOutputs;
    private List<float[]> policyInputs;
    private List<float[]> policyOutputs;

    private int epochCounter = 0;
    private int counter = 0;
    private int generationCounter = 0;
    private float startTime;

    void Start()
    {
        valueNetwork = new NeuralNetwork(valueShape, NeuralNetwork.ReLU, NeuralNetwork.Tanh, NeuralNetwork.ReLUDerivative, NeuralNetwork.TanhDerivative, NeuralNetwork.MSE, NeuralNetwork.MSEDerivative);
        policyNetwork = new NeuralNetwork(policyShape, NeuralNetwork.ReLU, NeuralNetwork.Softmax, NeuralNetwork.ReLUDerivative, NeuralNetwork.SoftmaxDerivative, NeuralNetwork.CategoricalCrossEntropy, NeuralNetwork.CategoricalCrossEntropyDerivative);

        generationCounter = startGeneration;
        int numGamesDone = startGeneration * numGames;
        if (startGeneration > 0) {
            valueNetwork.LoadNetwork(modelName+"-value" + numGamesDone.ToString());
            policyNetwork.LoadNetwork(modelName+"-policy" + numGamesDone.ToString());
        }
        valueInputs = new List<float[]>();
        valueOutputs = new List<float[]>();

        policyInputs = new List<float[]>();
        policyOutputs = new List<float[]>();

        startTime = Time.realtimeSinceStartup;
    }

    // Update is called once per frame
    void Update()
    {
        if (generationCounter < numGenerations)
        {
            generationText.text = "Generation " + (generationCounter + 1).ToString();
            if (counter < numGames)
            {
                secondaryDisplay.text = "Epoch 0: -";

                BoardNN board = new BoardNN();
                board.valueNetwork = valueNetwork;
                board.policyNetwork = policyNetwork;

                List<float[]> positions = new List<float[]>();
                BoardNN.Player winningPlayer;
                BoardNN.Player currentPlayer = BoardNN.Player.Red;
                while (true)
                {
                    float[] position = GetPosition(board.redBitboard, board.yellowBitboard);
                    winningPlayer = board.GetWinningPlayer();
                    if (winningPlayer != BoardNN.Player.None || board.IsFull())
                    {
                        break;
                    }

                    positions.Add(position);
                    policyInputs.Add(position);

                    if (currentPlayer == BoardNN.Player.Red)
                    {
                        BoardNN.MoveEval bestMove = board.TreeSearch(iterations, true);
                        board.MakeMove(bestMove.Move, currentPlayer);
                        currentPlayer = BoardNN.Player.Yellow;
                    }
                    else
                    {
                        BoardNN.MoveEval bestMove = board.TreeSearch(iterations, false);
                        board.MakeMove(bestMove.Move, currentPlayer);
                        currentPlayer = BoardNN.Player.Red;
                    }

                    policyOutputs.Add(board.promise);
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
            }
            else if (epochCounter < numEpochs)
            {
                float cost1 = valueNetwork.TrainOneEpoch(valueInputs, valueOutputs, 0.01f, 64);
                float cost2 = policyNetwork.TrainOneEpoch(policyInputs, policyOutputs, 0.01f, 64);
                // should be the same
                Debug.Log("Number of training samples: " + valueInputs.Count + " " + policyInputs.Count);
                epochCounter += 1;
                secondaryDisplay.text = "Epoch: " + epochCounter.ToString() + " Cost: " + cost1.ToString();
            }
            else
            {
                generationCounter += 1;
                int numGamesDone = generationCounter * numGames;
                valueNetwork.SaveNetwork(modelName+"-value" + numGamesDone.ToString());
                policyNetwork.SaveNetwork(modelName+"-policy" + numGamesDone.ToString());
                epochCounter = 0;
                counter = 0;
                startTime = Time.realtimeSinceStartup;
            }
        }
    }

    public float[] GetPosition(ulong redBitboard, ulong yellowBitboard)
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
}

public class BoardNN
{
    public enum Player { None, Red, Yellow };
    public enum NodeType
    {
        EXACT,
        LOWERBOUND,
        UPPERBOUND
    }

    public ulong redBitboard;
    public ulong yellowBitboard;
    public int nodes;
    public int[] heights;
    public int[] redHeights;
    public int[] yellowHeights;
    public Dictionary<ulong, TTEntry> tt;
    public EvaluationFunction evaluationFunction;
    public NeuralNetwork valueNetwork;
    public NeuralNetwork policyNetwork;

    public float[] redPositionalVals = new float[7] { 0, 1, 2, 3, 2, 1, 0 };
    public float[] yellowPositionVals = new float[7] { 1, 0, 1.5f, 3.5f, 1.5f, 0, 1 };

    public ulong zobristHash;
    private ulong[] zobristTable;
    public float maxDepth;
    public float[] promise;

    public BoardNN()
    {
        heights = new int[7] { 0, 0, 0, 0, 0, 0, 0 };
        promise = new float[7];
        InitiateBoard();
        InitializeZobrist();
    }

    public static float GenerateRandomNormal(float mean, float standardDeviation)
    {
        // Box-Muller transform
        float u1 = UnityEngine.Random.Range(0f, 1f); // Uniform(0,1] random doubles
        float u2 = UnityEngine.Random.Range(0f, 1f);
        float standardNormal = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Sin(2.0f * Mathf.PI * u2);

        // Scale and shift to match the specified mean and standard deviation
        return mean + standardDeviation * standardNormal;
    }

    // 0, 1, 2, .., 5 = first column
    public void InitiateBoard()
    {
        heights = new int[7] { 0, 0, 0, 0, 0, 0, 0 };
        redHeights = new int[7] { 0, 0, 0, 0, 0, 0, 0 };
        yellowHeights = new int[7] { 0, 0, 0, 0, 0, 0, 0 };

        redBitboard = 0;
        yellowBitboard = 0;
        tt = new Dictionary<ulong, TTEntry>();
    }

    public void ResetBoard()
    {
        redBitboard = 0;
        yellowBitboard = 0;
        heights = new int[7] { 0, 0, 0, 0, 0, 0, 0 };
        redHeights = new int[7] { 0, 0, 0, 0, 0, 0, 0 };
        yellowHeights = new int[7] { 0, 0, 0, 0, 0, 0, 0 };
        tt.Clear();
        zobristHash = 0;
    }

    public TreeNode GetRootNode(bool isRed) {
        State state = new State(redBitboard, yellowBitboard, heights);
        TreeNode rootNode = new TreeNode(valueNetwork, policyNetwork, state);
        rootNode.redToPlay = isRed;
        rootNode.depth = 0;
        rootNode.isRootNode = true;
        return rootNode;
    }

    public MoveEval BestMove(TreeNode rootNode) {
        TreeNode bestChild = null;
        float bestVisits = -1f;
    
        foreach (var child in rootNode.children)
        {
            if (child.N > bestVisits)
            {
                bestVisits = child.N;
                bestChild = child;
            }
        }

        maxDepth = rootNode.GetMaxDepth();
        // Return the best child’s column and Q as the chosen move:
        int bestMove = (bestChild != null) ? bestChild.priorMove : -1;
        float bestEval = (bestChild != null) ? bestChild.Q : 0;
        return new MoveEval(bestMove, bestEval);
    }

    public MoveEval TreeSearch(int iterations, bool isRed)
    {
        TreeNode rootNode = GetRootNode(isRed);

        // Run MCTS for the desired number of iterations:
        for (int i = 0; i < iterations; i++)
        {
            rootNode.Search();
        }

        // After MCTS finishes, pick the child with the highest visit count or best Q:
        TreeNode bestChild = null;
        float bestVisits = -1f;
        int totalVisits = 0;
    
        foreach (var child in rootNode.children)
        {
            // child.n should be an int but whatever
            totalVisits += (int)child.N;
            promise[child.priorMove] = (float)child.N;
            if (child.N > bestVisits)
            {
                bestVisits = child.N;
                bestChild = child;
            }
        }
        for (int i = 0; i < promise.Length; i++) {
            promise[i] /= (float)totalVisits;
        }

        maxDepth = rootNode.GetMaxDepth();
        // Return the best child’s column and Q as the chosen move:
        int bestMove = (bestChild != null) ? bestChild.priorMove : -1;
        float bestEval = (bestChild != null) ? bestChild.Q : 0;
        return new MoveEval(bestMove, bestEval);
    }


    public MoveEval Minimax(int depth, float alpha, float beta, bool isRed, int exploreFirst, float temperature)
    {
        nodes += 1;
        float alphaOriginal = alpha;
        Player winningPlayer = GetWinningPlayer();

        if (tt.TryGetValue(zobristHash, out TTEntry entry))
        {
            // Check if the stored depth is sufficient
            if (entry.Depth >= depth)
            {
                // Use node-type logic to adjust alpha/beta or just return
                switch (entry.Type)
                {
                    case NodeType.EXACT:
                        // It's an exact value for this depth
                        return new MoveEval(entry.BestMove, entry.Value);

                    case NodeType.LOWERBOUND:
                        // This value is a lower bound => effectively alpha = max(alpha, storedValue)
                        if (entry.Value > alpha) alpha = entry.Value;
                        break;

                    case NodeType.UPPERBOUND:
                        // This value is an upper bound => effectively beta = min(beta, storedValue)
                        if (entry.Value < beta) beta = entry.Value;
                        break;
                }

                // Alpha-beta cutoff check
                if (alpha >= beta)
                {
                    // We can prune here
                    return new MoveEval(entry.BestMove, entry.Value);
                }
            }
        }

        // Terminal nodes
        if (winningPlayer != Player.None)
        {
            if (winningPlayer == Player.Red)
            {
                float finalEval = (1000000 + depth) + UnityEngine.Random.Range(-0.01f, 0.01f);
                MoveEval bestMove = new MoveEval(-1, finalEval);

                // Store in TT
                StoreTT(finalEval, -1, depth, alphaOriginal, beta);
                // Higher depth means the win was reached earlier
                return bestMove;
            }
            else
            {
                float finalEval = -(1000000 + depth) + UnityEngine.Random.Range(-0.01f, 0.01f);
                MoveEval bestMove = new MoveEval(-1, finalEval);

                // Store in TT
                StoreTT(finalEval, -1, depth, alphaOriginal, beta);
                return bestMove;
            }
        }
        if (IsFull())
        {
            float finalEvaluation = UnityEngine.Random.Range(-0.01f, 0.01f);
            MoveEval bestMove = new MoveEval(-1, finalEvaluation);
            StoreTT(finalEvaluation, -1, depth, alphaOriginal, beta);
            return bestMove;
        }
        if (depth == 0)
        {
            // Heruistic
            float eval = HeuristicEvaluation() + GenerateRandomNormal(0, temperature);
            MoveEval bestMove = new MoveEval(-1, eval);
            StoreTT(eval, -1, depth, alphaOriginal, beta);
            return bestMove;
        }

        float bestValue = isRed ? Mathf.NegativeInfinity : Mathf.Infinity;
        int bestMove2 = 0;
        List<int> validMoves = GetValidMoves();
        List<int> moves = SortedMoves(validMoves, exploreFirst);

        if (isRed)
        {
            for (int i = 0; i < moves.Count; i++)
            {
                int column = moves[i];
                MakeMove(column, Player.Red);
                float eval = Minimax(depth - 1, alpha, beta, false, -1, temperature).Eval;
                if (eval > bestValue)
                {
                    bestValue = eval;
                    bestMove2 = column;
                }
                UnmakeMove(column, Player.Red);

                alpha = Mathf.Max(alpha, eval);
                if (alpha >= beta)
                {
                    break;
                }
            }
        }
        else
        {
            for (int i = 0; i < moves.Count; i++)
            {
                int column = moves[i];
                MakeMove(column, Player.Yellow);
                float eval = Minimax(depth - 1, alpha, beta, true, -1, temperature).Eval;
                if (eval < bestValue)
                {
                    bestMove2 = column;
                    bestValue = eval;
                }
                UnmakeMove(column, Player.Yellow);
                beta = Mathf.Min(beta, eval);
                if (beta <= alpha)
                {
                    break;
                }
            }
        }

        MoveEval bestME = new MoveEval(bestMove2, bestValue);
        StoreTT(bestValue, bestMove2, depth, alphaOriginal, beta);

        return bestME;
    }

    public void InitializeZobrist()
    {
        var random = new System.Random();
        zobristTable = new ulong[128];
        for (int i = 0; i < 128; i++)
        {
            ulong sixteenBits = (ulong)random.Next(1 << 16);
            ulong sixteenBits2 = (ulong)random.Next(1 << 16);
            ulong sixteenBits3 = (ulong)random.Next(1 << 16);
            ulong sixteenBits4 = (ulong)random.Next(1 << 16);
            ulong fullRange = (sixteenBits << 48) | (sixteenBits2 << 32) | (sixteenBits3 << 16) | (sixteenBits4);
            zobristTable[i] = fullRange;
        }
        zobristHash = 0;
    }

    public float HeuristicEvaluation()
    {
        return evaluationFunction(redBitboard, yellowBitboard);
    }

    public List<int> GetValidMoves()
    {
        List<int> moves = new List<int>();
        if (heights[3] < 6)
        {
            moves.Add(3);
        }
        for (int i = 1; i < 4; i++)
        {
            if (heights[3 + i] < 6)
            {
                moves.Add(3 + i);
            }
            if (heights[3 - i] < 6)
            {
                moves.Add(3 - i);
            }
        }
        return moves;
    }

    public List<int> SortedMoves(List<int> validMoves, int exploreFirst)
    {
        List<int> sortedMoves = new List<int>();
        if (exploreFirst != -1)
        {
            sortedMoves.Add(exploreFirst);
        }
        if (validMoves.Contains(3))
        {
            sortedMoves.Add(3);
        }
        for (int i = 1; i < 4; i++)
        {
            if (validMoves.Contains(3 + i))
            {
                sortedMoves.Add(3 + i);
            }
            if (validMoves.Contains(3 - i))
            {
                sortedMoves.Add(3 - i);
            }
        }
        return sortedMoves;
    }

    private bool HasConnectFour(ulong b)
    {
        // 1) Vertical check (shift by 1)
        {
            ulong m = b & (b >> 1);
            // If we can still find 2 more consecutive after that, there is a 4.
            if ((m & (m >> 2)) != 0UL)
                return true;
        }

        // 2) Horizontal check (shift by 6)
        {
            ulong m = b & (b >> 8);
            if ((m & (m >> 16)) != 0UL)
                return true;
        }

        // 3) Diagonal up-right (shift by 7)
        {
            ulong m = b & (b >> 9);
            if ((m & (m >> 18)) != 0UL)
                return true;
        }

        // 4) Diagonal up-left (shift by 5)
        {
            ulong m = b & (b >> 7);
            if ((m & (m >> 14)) != 0UL)
                return true;
        }

        return false;
    }


    public Player GetWinningPlayer()
    {
        if (HasConnectFour(redBitboard))
        {
            return Player.Red;
        }
        if (HasConnectFour(yellowBitboard))
        {
            return Player.Yellow;
        }
        return Player.None;
    }

    public Player GetBit(int position)
    {
        bool redVal = (redBitboard & ((ulong)1 << position)) != 0;
        if (redVal) return Player.Red;
        bool yellowVal = (yellowBitboard & ((ulong)1 << position)) != 0;
        if (yellowVal) return Player.Yellow;
        return Player.None;
    }

    public bool IsFull()
    {
        for (int i = 0; i < 7; i++)
        {
            if (heights[i] < 6)
            {
                return false;
            }
        }
        return true;
    }

    public void MakeMove(int column, Player player)
    {
        // First 6 = column 1 (row 1-6)
        // Next 6 = column 2 (row 1-6)
        // etc

        int bitPosition = 8 * column + heights[column];
        if (player == Player.Red)
        {
            redBitboard |= ((ulong)1 << bitPosition);
            zobristHash ^= zobristTable[2 * bitPosition];
            redHeights[column] += 1;
        }
        else
        {
            yellowBitboard |= ((ulong)1 << bitPosition);
            zobristHash ^= zobristTable[2 * bitPosition + 1];
            yellowHeights[column] += 1;
        }
        heights[column] += 1;
    }

    public void UnmakeMove(int column, Player player)
    {
        int bitPosition = 8 * column + heights[column] - 1;
        if (player == Player.Red)
        {
            redBitboard = redBitboard & ~((ulong)1 << bitPosition);
            zobristHash ^= zobristTable[2 * bitPosition];
            redHeights[column] -= 1;
        }
        else
        {
            yellowBitboard = yellowBitboard & ~((ulong)1 << bitPosition);
            zobristHash ^= zobristTable[2 * bitPosition + 1];
            yellowHeights[column] -= 1;
        }
        heights[column] -= 1;
    }

    public bool IsValidMove(int column)
    {
        return heights[column] < 6;
    }

    public static int CountBits(ulong value)
    {
        int count = 0;
        while (value != 0)
        {
            count++;
            value &= value - 1;
        }
        return count;
    }

    public struct MoveEval
    {
        public MoveEval(int move, float eval)
        {
            Move = move;
            Eval = eval;
        }

        public int Move { get; }
        public float Eval { get; }
    }

    private void StoreTT(float value, int bestMove, int depth, float alphaOriginal, float beta)
    {
        NodeType nodeType;

        // If the eval is <= alphaOriginal, then we had an "upper bound" case
        if (value <= alphaOriginal)
        {
            nodeType = NodeType.UPPERBOUND;
        }
        // If the eval is >= beta, then we had a "lower bound" case
        else if (value >= beta)
        {
            nodeType = NodeType.LOWERBOUND;
        }
        // Otherwise it’s an exact value
        else
        {
            nodeType = NodeType.EXACT;
        }

        TTEntry entry = new TTEntry(value, bestMove, depth, nodeType);
        tt[zobristHash] = entry;
    }

    public struct TTEntry
    {
        public float Value;      // The evaluation score
        public int BestMove;     // The move that led to Value
        public int Depth;        // The depth at which this was computed
        public NodeType Type;    // EXACT, LOWERBOUND, or UPPERBOUND

        public TTEntry(float value, int bestMove, int depth, NodeType type)
        {
            Value = value;
            BestMove = bestMove;
            Depth = depth;
            Type = type;
        }
    }
}

