using UnityEngine;
using System.Collections.Generic;
using System;
using System.Threading.Tasks;
using System.Threading;
using System.Linq;

public class BoardNN : IDisposable
{
    private bool disposed = false;
    private TreeNode currentRootNode;

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
    public NeuralNetwork valueNetwork;
    public NeuralNetwork policyNetwork;
    public float rootNoise;
    public float dirichletAlpha;

    public float maxDepth;
    public float[] promise;

    private readonly ParallelOptions parallelOptions = new ParallelOptions {
        MaxDegreeOfParallelism = System.Environment.ProcessorCount // Use all available cores
    };

    public BoardNN()
    {
        heights = new int[7] { 0, 0, 0, 0, 0, 0, 0 };
        promise = new float[7];
        InitiateBoard();
    }

    // 0, 1, 2, .., 5 = first column
    public void InitiateBoard()
    {
        heights = new int[7] { 0, 0, 0, 0, 0, 0, 0 };

        redBitboard = 0;
        yellowBitboard = 0;
    }

    public void ResetBoard()
    {
        redBitboard = 0;
        yellowBitboard = 0;
        heights = new int[7] { 0, 0, 0, 0, 0, 0, 0 };
    }

    public TreeNode GetRootNode(bool isRed) {
        // Dispose of previous root node if it exists
        if (currentRootNode != null)
        {
            currentRootNode.Dispose();
            currentRootNode = null;
        }

        State state = new State(redBitboard, yellowBitboard, heights);
        currentRootNode = new TreeNode(valueNetwork, policyNetwork, state);
        currentRootNode.redToPlay = isRed;
        currentRootNode.depth = 0;
        return currentRootNode;
    }

    public MoveEval BestMove(TreeNode rootNode, float temperature = 0f) {
        if (temperature > 0) {
            // Calculate probabilities based on visit counts
            float[] probs = new float[7];
            float sum = 0;
            foreach (var child in rootNode.children) {
                probs[child.priorMove] = Mathf.Pow(child.N, 1/temperature);
                sum += probs[child.priorMove];
            }
            
            // Normalize
            for (int i = 0; i < 7; i++) {
                probs[i] /= sum;
            }
            
            // Select move based on probabilities
            float r = UnityEngine.Random.value;
            float cumulativeProb = 0;
            foreach (var child in rootNode.children) {
                cumulativeProb += probs[child.priorMove];
                if (r <= cumulativeProb) {
                    return new MoveEval(child.priorMove, child.Q);
                }
            }
        }

        // Default to deterministic selection (temperature = 0)
        TreeNode bestChild = null;
        float bestVisits = -1.0f;
        float totalVisits = 0;

        for (int i = 0; i < 7; i++) {
            promise[i] = 0;
        }

        foreach (var child in rootNode.children) {
            promise[child.priorMove] = child.N;
            totalVisits += child.N;
            if (child.N > bestVisits) {
                bestVisits = child.N;
                bestChild = child;
            }
        }

        for (int i = 0; i < 7; i++) {
            promise[i] /= totalVisits;
        }

        maxDepth = rootNode.GetMaxDepth();
        int bestMove = (bestChild != null) ? bestChild.priorMove : -1;
        float bestEval = (bestChild != null) ? bestChild.Q : 0;
        return new MoveEval(bestMove, bestEval);
    }

    public MoveEval TreeSearch(int iterations, bool isRed, float temperature = 1.0f)
    {
        TreeNode rootNode = GetRootNode(isRed);
        
        // Create a counter for progress tracking
        int completedIterations = 0;

        try {
            Parallel.For(0, iterations, parallelOptions, i => {
                rootNode.Search(rootNoise, dirichletAlpha);
                Interlocked.Increment(ref completedIterations);
            });
        }
        catch (AggregateException ae) {
            Debug.LogError($"Parallel search failed: {ae.Message}");
            foreach (var e in ae.InnerExceptions) {
                Debug.LogError($"Inner exception: {e.Message}");
            }
        }

        // Calculate promises (visit counts)
        TreeNode bestChild = null;
        float bestVisits = -1f;
        float totalVisits = 0f;
    
        // Zero out promises first
        for (int i = 0; i < 7; i++) {
            promise[i] = 0f;
        }

        foreach (var child in rootNode.children) {
            totalVisits += (float)child.N;
            promise[child.priorMove] = (float)child.N;
            if (child.N > bestVisits) {
                bestVisits = child.N;
                bestChild = child;
            }
        }
        // Normalize promises
        for (int i = 0; i < 7; i++) {
            promise[i] /= (float)totalVisits;
        }

        // Apply temperature scaling to visit counts
        float[] probs = new float[7];
        float sum = 0;
        for (int i = 0; i < 7; i++) {
            probs[i] = 0;
        }
        for (int i = 0; i < rootNode.children.Count; i++) {
            probs[rootNode.children[i].priorMove] = Mathf.Pow(rootNode.children[i].N, 1/temperature);
            sum += probs[rootNode.children[i].priorMove];
        }
        
        // Normalize
        for (int i = 0; i < 7; i++) {
            probs[i] /= sum;
        }
        
        // During training, sometimes choose moves proportionally
        if (temperature > 0) {
            float r = UnityEngine.Random.value;
            float cumulativeProb = 0;
            for (int i = 0; i < rootNode.children.Count; i++) {
                cumulativeProb += probs[rootNode.children[i].priorMove];
                if (r <= cumulativeProb) {
                    return new MoveEval(rootNode.children[i].priorMove, rootNode.children[i].Q);
                }
            }
        }        

        int bestMove = (bestChild != null) ? bestChild.priorMove : -1;
        float bestEval = (bestChild != null) ? bestChild.Q : 0;
        return new MoveEval(bestMove, bestEval);
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
        }
        else
        {
            yellowBitboard |= ((ulong)1 << bitPosition);
        }
        heights[column] += 1;
    }

    public void UnmakeMove(int column, Player player)
    {
        int bitPosition = 8 * column + heights[column] - 1;
        if (player == Player.Red)
        {
            redBitboard = redBitboard & ~((ulong)1 << bitPosition);
        }
        else
        {
            yellowBitboard = yellowBitboard & ~((ulong)1 << bitPosition);
        }
        heights[column] -= 1;
    }

    public bool IsValidMove(int column)
    {
        return heights[column] < 6;
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

    public void ShowBoard()
    {
        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        sb.AppendLine("Current Board State:");
        sb.AppendLine("-----------------------------");
        
        for (int row = 5; row >= 0; row--)
        {
            string currentRow = "|";
            for (int col = 0; col < 7; col++)
            {
                int position = 8 * col + row;
                bool isRed = (redBitboard & ((ulong)1 << position)) != 0;
                bool isYellow = (yellowBitboard & ((ulong)1 << position)) != 0;
                
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

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!disposed)
        {
            if (disposing)
            {
                currentRootNode?.Dispose();
                currentRootNode = null;
                
                // Clear references but don't dispose shared networks
                valueNetwork = null;
                policyNetwork = null;
            }
            disposed = true;
        }
    }

    ~BoardNN()
    {
        Dispose(false);
    }
}