using UnityEngine;
using System.Collections.Generic;
using System;
using System.Threading.Tasks;
using System.Threading;
using System.Linq;
using System.Collections.Concurrent;

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


public class TreeNode : IDisposable {
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

    // For demonstration, we'll define a constant for exploration:
    private const float C_PUCT = 1.4f;

    // Add lock object for thread safety
    private readonly object lockObject = new object();
    private readonly object _networkLock = new object();

    // Add a thread-safe random number generator
    private static readonly ThreadLocal<System.Random> random = 
        new ThreadLocal<System.Random>(() => new System.Random(Interlocked.Increment(ref randomSeed)));
    private static int randomSeed = Environment.TickCount ^ Guid.NewGuid().GetHashCode();

    private const float EARLY_EXPLORE_FACTOR = 2.0f;
    private const int EARLY_GAME_VISITS = 0;
    private const int MIN_VISITS_BEFORE_COMMIT = 0;
    private static bool useExplorationConstraints = true;

    // Add cache size limits
    private const int MAX_CACHE_SIZE = 1000000; // Adjust this value based on your memory constraints
    private static readonly object cacheLockObject = new object();

    // Replace single transposition table with separate ones for policy and value
    public static readonly ConcurrentDictionary<long, float[]> policyCache 
        = new ConcurrentDictionary<long, float[]>();
    public static readonly ConcurrentDictionary<long, float> valueCache 
        = new ConcurrentDictionary<long, float>();

    // Add static flag for evaluation mode
    private static bool useUniformEvaluation = false;
    private static readonly object evaluationModeLock = new object();

    private bool disposed = false;

    // Add method to compute hash
    private static long ComputeHash(State state) {
        return state.redBitboard.GetHashCode() ^ (state.yellowBitboard.GetHashCode() * 31L);
    }

    public TreeNode(NeuralNetwork valueNetwork, NeuralNetwork policyNetwork, State state) {
        this.valueNetwork = valueNetwork;
        this.policyNetwork = policyNetwork;
        this.state = state;
        visits = 0;
        isRootNode = false;
    }

    public void CreateChildren() {
        List<int> possibleMoves = GetValidMoves();
        children = new List<TreeNode>();

        float[] policy = EvaluatePolicy(state);
        float truePolicyVal = 0f;

        // Add validation for NaN values
        for (int i = 0; i < possibleMoves.Count; i++) {
            int move = possibleMoves[i];
            truePolicyVal += policy[move];
        }

        for (int i = 0; i < possibleMoves.Count; i++)
        {
            int move = possibleMoves[i];
            State childState = MakeMove(move);
            TreeNode child = new TreeNode(valueNetwork, policyNetwork, childState);
            child.priorMove = move;
            child.redToPlay = !redToPlay;
            child.depth = depth + 1;

            // Absolute eval
            if (HasConnectFour(childState.redBitboard))
            {
                child.isTerminal = true;
                child.eval = 1.0f * Mathf.Pow(0.99f, child.state.heights.Sum());
            }
            else if (HasConnectFour(childState.yellowBitboard))
            {
                child.isTerminal = true;
                child.eval = -1.0f * Mathf.Pow(0.99f, child.state.heights.Sum());
            }
            else if (IsFull(childState))
            {
                child.isTerminal = true;
                child.eval = 0f;
            }
            else
            {
                float rawEval = EvaluateValue(childState);
                child.eval = rawEval;
            }

            child.prior = policy[move] / (float)truePolicyVal;
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
        float[] result = new float[2 * 6 * 7 + 1];  // 85 elements total (84 board state + 1 parity)

        int totalPieces = 0;
        for (int row = 0; row < 6; row++)
        {
            for (int col = 0; col < 7; col++)
            {
                int bitPosition = 8 * col + row;
                
                bool isRed = (state.redBitboard & ((ulong)1 << bitPosition)) != 0;
                bool isYellow = (state.yellowBitboard & ((ulong)1 << bitPosition)) != 0;
                
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

    public int GetMinDepth() {
        if (children == null)
        {
            return depth;
        }
        int minDepth = 10000;
        for (int i = 0; i < children.Count; i++)
        {
            minDepth = Mathf.Min(minDepth, children[i].GetMinDepth());
        }
        return minDepth;
    }

    public void CreateChildrenWithRootNoise(float alpha, float epsilon)
    {
        CreateChildren();

        // Only apply noise if there's more than one child
        if (alpha > 0 && children.Count > 1) {
            // Now blend in Dirichlet noise at the root:
            float[] noise = NoiseUtils.SampleDirichlet(children.Count, alpha);

            for (int i = 0; i < children.Count; i++)
            {
                float oldPrior = children[i].prior;  // e.g. from NN
                float newPrior = (1 - epsilon) * oldPrior + epsilon * noise[i];
                children[i].prior = newPrior;
            }
        }
    }

    public float Search(float rootNoise, float dirichletAlpha)
    {
        if (isTerminal) {
            lock(lockObject) {
                N += 1;
                W += eval ?? 0f;  // Use null coalescing operator
                Q = W / N;
                return eval ?? 0f;
            }
        }

        // First, ensure children are created under a lock
        lock(lockObject) {
            if (children == null) {
                CreateChildrenWithRootNoise(rootNoise, dirichletAlpha);
                
                if (children.Count == 0) {
                    N += 1;
                    return 0;
                }

                // Initialize all children at once
                float initialValue = EvaluateValue(state);
                
                foreach (var child in children) {
                    if (child.isTerminal) {
                        child.N = 1;
                        child.W = child.eval ?? 0f;
                        child.Q = child.W;
                    }
                }

                W += initialValue;
                N += 1;
                Q = W / N;
                return initialValue;
            }
        }

        // Take a snapshot of total visits under lock
        float totalVisits;
        lock(lockObject) {
            totalVisits = children.Sum(c => c.N);
        }

        // Select best child using our thread-safe selection
        TreeNode bestChild = SelectBestChild(totalVisits);
        if (bestChild == null) {
            Debug.LogError($"No best child found. Children count: {children?.Count ?? 0}");
            return 0;
        }

        // Perform the recursive search
        float childValue = bestChild.Search(rootNoise, dirichletAlpha);

        // Update statistics under lock
        lock(lockObject) {
            W += childValue;
            N += 1;
            Q = W / N;
        }

        return childValue;
    }

    private TreeNode SelectBestChild(float totalVisits)
    {
        TreeNode bestChild = null;
        float bestScore = float.NegativeInfinity;
        
        // Take a single snapshot under one lock instead of multiple nested locks
        List<(TreeNode child, float N, float Q, float prior, bool needsExploration)> childStats;
        lock(lockObject) {
            childStats = children.Select(child => (
                child,
                child.N,
                redToPlay ? child.Q : -child.Q,
                child.prior,
                child.N < MIN_VISITS_BEFORE_COMMIT
            )).ToList();
        }

        // First, check if any child needs minimum exploration
        var unexploredChildren = childStats.Where(c => useExplorationConstraints && c.needsExploration).ToList();
        if (unexploredChildren.Any()) {
            return unexploredChildren.OrderBy(c => c.N).First().child;
        }

        // Normal UCT selection
        foreach (var (child, childN, childQ, childPrior, _) in childStats) {
            float uct = GetUCTScore(childN, childQ, childPrior, totalVisits);
            if (uct > bestScore) {
                bestScore = uct;
                bestChild = child;
            }
        }

        return bestChild;
    }

    private float[] EvaluatePolicy(State state) {
        // Check if using uniform evaluation
        lock(evaluationModeLock) {
            if (useUniformEvaluation) {
                float[] uniformPolicy = new float[7];
                List<int> validMoves = GetValidMoves();
                float uniformValue = 1.0f / validMoves.Count;
                foreach (int move in validMoves) {
                    uniformPolicy[move] = uniformValue;
                }
                return uniformPolicy;
            }
        }

        // Original neural network evaluation
        long hash = ComputeHash(state);
        
        var result = policyCache.GetOrAdd(hash, _ => {
            lock(_networkLock) {
                if (policyCache.Count > MAX_CACHE_SIZE) {
                    TrimCache(policyCache);
                }
                return policyNetwork.Forward(StateToInput(state));
            }
        });
        
        return result;
    }
    
    private float EvaluateValue(State state) {
        // Check if using uniform evaluation
        lock(evaluationModeLock) {
            if (useUniformEvaluation) {
                return 0.0f;
            }
        }

        // Original neural network evaluation
        long hash = ComputeHash(state);
        
        var result = valueCache.GetOrAdd(hash, _ => {
            lock(_networkLock) {
                if (valueCache.Count > MAX_CACHE_SIZE) {
                    TrimCache(valueCache);
                }
                return valueNetwork.Forward(StateToInput(state))[0];
            }
        });
            
        return result;
    }

    private float GetUCTScore(float childN, float childQ, float childPrior, float totalVisits)
    {
        // Ensure minimum exploration only if constraints are enabled
        if (useExplorationConstraints && childN < MIN_VISITS_BEFORE_COMMIT) {
            return float.MaxValue - (MIN_VISITS_BEFORE_COMMIT - childN);
        }

        float explorationTerm;
        if (useExplorationConstraints && totalVisits < EARLY_GAME_VISITS) {
            explorationTerm = C_PUCT * EARLY_EXPLORE_FACTOR * childPrior * Mathf.Sqrt(totalVisits + 1);
        } else {
            explorationTerm = C_PUCT * childPrior * Mathf.Sqrt(totalVisits + 1);
        }

        // Smooth early evaluations only if constraints are enabled
        float visitTerm = 1 + childN + 1e-6f;
        float exploitationTerm = useExplorationConstraints && childN < EARLY_GAME_VISITS ? 
            childQ * (childN / EARLY_GAME_VISITS) : childQ;
        
        return exploitationTerm + explorationTerm / visitTerm;
    }

    // Add a public method to toggle the exploration constraints
    public static void SetExplorationConstraints(bool enabled)
    {
        useExplorationConstraints = enabled;
    }

    // Update clear method to handle both caches
    public static void ClearTranspositionTable() {
        try {
            // Clear policy cache
            if (policyCache != null) {
                var oldPolicyCache = policyCache;
                policyCache.Clear();
                foreach (var kvp in oldPolicyCache) {
                    oldPolicyCache.TryRemove(kvp.Key, out _);
                }
            }

            // Clear value cache
            if (valueCache != null) {
                var oldValueCache = valueCache;
                valueCache.Clear();
                foreach (var kvp in oldValueCache) {
                    oldValueCache.TryRemove(kvp.Key, out _);
                }
            }

            // Trim memory more aggressively
            GC.Collect(2, GCCollectionMode.Forced, true);
            GC.WaitForPendingFinalizers();
            GC.Collect(2, GCCollectionMode.Forced, true);
        }
        catch (Exception e) {
            Debug.LogError($"Error clearing transposition table: {e.Message}\n{e.StackTrace}");
        }
    }

    // Add a method to monitor cache sizes
    public static void LogCacheStats() {
        long policyCacheSize = policyCache?.Count ?? 0;
        long valueCacheSize = valueCache?.Count ?? 0;
        Debug.Log($"Cache sizes - Policy: {policyCacheSize}, Value: {valueCacheSize}");
    }

    private static void TrimCache<T>(ConcurrentDictionary<long, T> cache)
    {
        if (cache.Count > MAX_CACHE_SIZE)
        {
            lock (cacheLockObject)
            {
                // Remove ~20% of entries when we hit the limit
                int entriesToRemove = MAX_CACHE_SIZE / 5;
                var keysToRemove = cache.Keys.Take(entriesToRemove).ToList();
                foreach (var key in keysToRemove)
                {
                    cache.TryRemove(key, out _);
                }
            }
        }
    }

    // Add public method to toggle evaluation mode
    public static void SetUniformEvaluation(bool enabled) {
        lock(evaluationModeLock) {
            useUniformEvaluation = enabled;
        }
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
                if (children != null)
                {
                    foreach (var child in children)
                    {
                        child?.Dispose();
                    }
                    children.Clear();
                    children = null;
                }

                // Clear references
                valueNetwork = null;
                policyNetwork = null;
                state = default; // This is a struct, so it will be cleared
            }
            disposed = true;
        }
    }

    // Add destructor
    ~TreeNode()
    {
        Dispose(false);
    }
}
