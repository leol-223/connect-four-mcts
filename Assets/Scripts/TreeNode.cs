using UnityEngine;
using System.Collections.Generic;
using System;
using System.Threading.Tasks;
using System.Threading;
using System.Linq;

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

    // Add lock object for thread safety
    private readonly object lockObject = new object();

    // Add a thread-safe random number generator
    private static readonly ThreadLocal<System.Random> random = 
        new ThreadLocal<System.Random>(() => new System.Random(Interlocked.Increment(ref randomSeed)));
    private static int randomSeed = Environment.TickCount;

    // Replace UnityEngine.Random.Range with this method
    private float RandomRange(float min, float max) {
        return (float)(random.Value.NextDouble() * (max - min) + min);
    }

    public static float GenerateRandomNormal(float mean, float standardDeviation)
    {
        // Box-Muller transform with thread-safe random
        float u1 = (float)random.Value.NextDouble(); // Uniform(0,1] random doubles
        float u2 = (float)random.Value.NextDouble();
        float standardNormal = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Sin(2.0f * Mathf.PI * u2);

        // Scale and shift to match the specified mean and standard deviation
        return mean + standardDeviation * standardNormal;
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

        float[] policy = policyNetwork.Evaluate(StateToInput(state));
        // priors should add up to 1
        // If the policy network is stupid (ie assigns 1/7 to everything) but not every move is possible then we've got a problem
        float truePolicyVal = 0f;

        for (int i = 0; i < possibleMoves.Count; i++)
        {
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

            child.prior = policy[move] / truePolicyVal;
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
        float[] result = new float[84];

        // Match the exact same position calculation as ShowBoard
        for (int col = 0; col < 7; col++)
        {
            for (int row = 0; row < 6; row++)
            {
                int bitPosition = 8 * col + row;
                int nnPosition = col * 6 + row;
                
                bool isRed = (state.redBitboard & ((ulong)1 << bitPosition)) != 0;
                bool isYellow = (state.yellowBitboard & ((ulong)1 << bitPosition)) != 0;
                
                result[nnPosition] = isRed ? 1.0f : 0.0f;
                result[nnPosition + 42] = isYellow ? 1.0f : 0.0f;
            }
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
        CreateChildren(); // sets child.prior in some default way
                          // e.g. from NN policy or uniform

        if (alpha > 0) {
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
                W += (float)eval;
                Q = W / N;
                return (float)eval;
            }
        }

        if (children == null) {
            lock(lockObject) {
                // Double-check locking pattern
                if (children == null) {
                    if (isRootNode) {
                        CreateChildren();
                    } else {
                        CreateChildrenWithRootNoise(rootNoise, dirichletAlpha);
                    }

                    foreach (var child in children) {
                        if (child.isTerminal) {
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
            }
        }

        float totalVisits;
        lock(lockObject) {
            totalVisits = children.Sum(c => c.N);
        }

        TreeNode bestChild = null;
        float bestScore = float.NegativeInfinity;

        // Add check for empty children list
        if (children.Count == 0) {
            lock(lockObject) {
                W += 0;  // Draw value for a terminal state
                N += 1;
                Q = W / N;
                return 0;
            }
        }

        foreach (var c in children) {
            float uct;
            float childN, childQ, childPrior;
            
            lock(c.lockObject) {
                childN = c.N;
                childQ = c.Q;
                childPrior = c.prior;
            }

            if (childN == 0) {
                uct = c.isTerminal ? ((float)c.eval) : (C_PUCT * childPrior * Mathf.Sqrt(totalVisits + 1));
            } else {
                uct = C_PUCT * childPrior * Mathf.Sqrt(totalVisits + 1) / (1 + childN) + childQ;
            }
            
            if (uct > bestScore) {
                bestScore = uct;
                bestChild = c;
            }
        }

        // Add null check
        if (bestChild == null) {
            Debug.LogError("No best child found - this shouldn't happen if children list is non-empty");
            lock(lockObject) {
                W += 0;
                N += 1;
                Q = W / N;
                return 0;
            }
        }

        float childValue = -bestChild.Search(rootNoise, dirichletAlpha);

        lock(lockObject) {
            W += childValue;
            N += 1;
            Q = W / N;
        }

        return childValue;
    }

}