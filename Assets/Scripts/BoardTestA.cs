using UnityEngine;
using System.Collections.Generic;
using System;
using System.Numerics;


public class BoardTestA
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

    public float[] redPositionalVals = new float[7] { 0, 1, 2, 3, 2, 1, 0 };
    public float[] yellowPositionVals = new float[7] { 0, 1, 2, 3, 2, 1, 0 };

    public ulong zobristHash;
    private ulong[] zobristTable;

    public BoardTestA()
    {
        heights = new int[7] { 0, 0, 0, 0, 0, 0, 0 };
        InitiateBoard();
        InitializeZobrist();
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

    public MoveEval GetBestMove(int depth, float alpha, float beta, bool isRed, int exploreFirst)
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
            float eval = HeuristicEvaluation() + UnityEngine.Random.Range(-0.01f, 0.01f);
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
                float eval = GetBestMove(depth - 1, alpha, beta, false, -1).Eval;
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
                float eval = GetBestMove(depth - 1, alpha, beta, true, -1).Eval;
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
        float evaluation = 0f;
        for (int i = 0; i < 7; i++)
        {
            evaluation += redHeights[i] * redPositionalVals[i];
            evaluation -= yellowHeights[i] * yellowPositionVals[i];
        }
        return evaluation;
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
