using UnityEngine;
using System.Collections.Generic;
using System;

public class BoardTestA
{
    public enum Player { None, Red, Yellow };
    public enum NodeType
    {
        EXACT,
        LOWERBOUND,
        UPPERBOUND
    }

    public Player[] board;
    public int[] heights;
    public int nodes;
    public int ttnodes;
    public Dictionary<ulong, TTEntry> tt;

    public ulong zobristHash;
    private ulong[] zobristTable;
    private int[] redCounts;
    private int[] yellowCounts;

    private int[] redRowCounts;
    private int[] yellowRowCounts;

    public BoardTestA()
    {
        InitiateBoard();
        InitializeZobrist();
    }

    // 0, 1, 2, .., 5 = first column
    public void InitiateBoard()
    {
        board = new Player[42];
        heights = new int[7];
        redCounts = new int[7];
        yellowCounts = new int[7];
        redRowCounts = new int[6];
        yellowRowCounts = new int[6];
        for (int i = 0; i < 7; i++)
        {
            heights[i] = 0;
            redCounts[i] = 0;
            yellowCounts[i] = 0;
            if (i != 6)
            {
                redRowCounts[i] = 0;
                yellowRowCounts[i] = 0;
            }
        }
        for (int i = 0; i < 42; i++)
        {
            board[i] = Player.None;
        }
        tt = new Dictionary<ulong, TTEntry>();
    }

    public void ResetBoard()
    {
        for (int i = 0; i < 7; i++)
        {
            heights[i] = 0;
            redCounts[i] = 0;
            yellowCounts[i] = 0;
            if (i != 6)
            {
                redRowCounts[i] = 0;
                yellowRowCounts[i] = 0;
            }
        }
        for (int i = 0; i < 42; i++)
        {
            board[i] = Player.None;
        }
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
                float finalEval = (1000000 + depth) + UnityEngine.Random.Range(-1f, 1f) * 0.01f;
                MoveEval bestMove = new MoveEval(-1, finalEval);

                // Store in TT
                StoreTT(finalEval, -1, depth, alphaOriginal, beta);
                // Higher depth means the win was reached earlier
                return bestMove;
            }
            else
            {
                float finalEval = -(1000000 + depth) + UnityEngine.Random.Range(-1f, 1f) * 0.01f;
                MoveEval bestMove = new MoveEval(-1, finalEval);

                // Store in TT
                StoreTT(finalEval, -1, depth, alphaOriginal, beta);
                return bestMove;
            }
        }
        if (IsFull())
        {
            // Heuristic
            float finalEvaluation = UnityEngine.Random.Range(-1f, 1f) * 0.01f;
            MoveEval bestMove = new MoveEval(-1, finalEvaluation);
            StoreTT(finalEvaluation, -1, depth, alphaOriginal, beta);
            return bestMove;
        }
        if (depth == 0)
        {
            float eval = HeuristicEvaluation();
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
        zobristTable = new ulong[84];
        for (int i = 0; i < 84; i++)
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
        float evaluation = 0;
        // Random jiggle
        evaluation += UnityEngine.Random.Range(-1f, 1f) * 0.01f;

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


    public Player GetWinningPlayer()
    {
        for (int i = 0; i < 7; i++)
        {
            for (int j = 0; j < heights[i] - 3; j++)
            {
                Player endingValue = board[6 * i + j + 3];
                bool connectFour = true;
                for (int k = 0; k < 3; k++)
                {
                    if (board[6 * i + j + k] != endingValue)
                    {
                        connectFour = false;
                        break;
                    }
                }
                if (connectFour)
                {
                    return endingValue;
                }
            }
        }
        // Rows
        for (int j = 0; j < 6; j++)
        {
            for (int i = 0; i < 4; i++)
            {
                Player endingValue = board[6 * (i + 3) + j];
                if (endingValue == Player.None)
                {
                    break;
                }
                bool connectFour = true;
                for (int k = 0; k < 3; k++)
                {
                    if (board[6 * (i + k) + j] != endingValue)
                    {
                        connectFour = false;
                        break;
                    }
                }
                if (connectFour)
                {
                    return endingValue;
                }
            }
        }
        // Diagonals going down-right
        for (int i = 0; i < 4; i++)
        {
            for (int j = 3; j < 6; j++)
            {
                // Top left value
                Player endingValue = board[6 * i + j];
                bool connectFour = true;
                for (int k = 1; k < 4; k++)
                {
                    if (board[6 * (i + k) + (j - k)] != endingValue)
                    {
                        connectFour = false;
                        break;
                    }
                }
                if (connectFour)
                {
                    return endingValue;
                }
            }
        }
        // Diagonals going down-left
        for (int i = 3; i < 7; i++)
        {
            for (int j = 3; j < 6; j++)
            {
                // Top right value
                Player endingValue = board[6 * i + j];
                bool connectFour = true;
                for (int k = 1; k < 4; k++)
                {
                    if (board[6 * (i - k) + (j - k)] != endingValue)
                    {
                        connectFour = false;
                        break;
                    }
                }
                if (connectFour)
                {
                    return endingValue;
                }
            }
        }
        return Player.None;
    }
    /*
    public float WindowScore()
    {
        float score = 0;

        for (int i = 0; i < 7; i++)
        {
            for (int j = 0; j < heights[i] - 3; j++)
            {
                int numEmpty = 0;
                int numRed = 0;
                int numYellow = 0;
                for (int k = 0; k < 4; k++)
                {
                    Player val = board[6 * i + j + k];
                    if (val == Player.None)
                    {
                        numEmpty += 1;
                    }
                    else if (val == Player.Red)
                    {
                        numRed += 1;
                    }
                    else
                    {
                        numYellow += 1;
                    }
                }
                if (numRed > 1 && numYellow == 0)
                {
                    if (numRed == 3)
                    {
                        score += windowScore3;
                    }
                    else
                    {
                        score += windowScore2;
                    }
                }
                if (numYellow > 1 && numRed == 0)
                {
                    if (numYellow == 3)
                    {
                        score -= windowScore3;
                    }
                    else
                    {
                        score -= windowScore2;
                    }
                }
            }
        }
        // Rows
        for (int j = 0; j < 6; j++)
        {
            for (int i = 0; i < 4; i++)
            {
                int numEmpty = 0;
                int numRed = 0;
                int numYellow = 0;

                for (int k = 0; k < 4; k++)
                {
                    Player val = board[6 * (i + k) + j];
                    if (val == Player.None)
                    {
                        numEmpty += 1;
                    }
                    else if (val == Player.Red)
                    {
                        numRed += 1;
                    }
                    else
                    {
                        numYellow += 1;
                    }
                }
                if (numRed > 1 && numYellow == 0)
                {
                    if (numRed == 3)
                    {
                        score += windowScore3;
                    }
                    else
                    {
                        score += windowScore2;
                    }
                }
                if (numYellow > 1 && numRed == 0)
                {
                    if (numYellow == 3)
                    {
                        score -= windowScore3;
                    }
                    else
                    {
                        score -= windowScore2;
                    }
                }
            }
        }
        // Diagonals going down-right
        for (int i = 0; i < 4; i++)
        {
            for (int j = 3; j < 6; j++)
            {
                int numEmpty = 0;
                int numRed = 0;
                int numYellow = 0;
                // Top left value
                for (int k = 0; k < 4; k++)
                {
                    Player val = board[6 * (i + k) + (j - k)];
                    if (val == Player.None)
                    {
                        numEmpty += 1;
                    }
                    else if (val == Player.Red)
                    {
                        numRed += 1;
                    }
                    else
                    {
                        numYellow += 1;
                    }
                }
                if (numRed > 1 && numYellow == 0)
                {
                    if (numRed == 3)
                    {
                        score += windowScore3;
                    }
                    else
                    {
                        score += windowScore2;
                    }
                }
                if (numYellow > 1 && numRed == 0)
                {
                    if (numYellow == 3)
                    {
                        score -= windowScore3;
                    }
                    else
                    {
                        score -= windowScore2;
                    }
                }
            }
        }
        // Diagonals going down-left
        for (int i = 3; i < 7; i++)
        {
            for (int j = 3; j < 6; j++)
            {
                int numEmpty = 0;
                int numRed = 0;
                int numYellow = 0;
                // Top right value
                for (int k = 0; k < 4; k++)
                {
                    Player val = board[6 * (i - k) + (j - k)];
                    if (val == Player.None)
                    {
                        numEmpty += 1;
                    }
                    else if (val == Player.Red)
                    {
                        numRed += 1;
                    }
                    else
                    {
                        numYellow += 1;
                    }
                }
                if (numRed > 1 && numYellow == 0)
                {
                    if (numRed == 3)
                    {
                        score += windowScore3;
                    }
                    else
                    {
                        score += windowScore2;
                    }
                }
                if (numYellow > 1 && numRed == 0)
                {
                    if (numYellow == 3)
                    {
                        score -= windowScore3;
                    }
                    else
                    {
                        score -= windowScore2;
                    }
                }
            }
        }
        return score;
    }
    */

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
        if (player == Player.Red)
        {
            redCounts[column] += 1;
            redRowCounts[heights[column]] += 1;
            zobristHash ^= zobristTable[2 * (6 * column + heights[column])];
        }
        else
        {
            yellowCounts[column] += 1;
            yellowRowCounts[heights[column]] += 1;
            zobristHash ^= zobristTable[2 * (6 * column + heights[column]) + 1];
        }
        board[6 * column + heights[column]] = player;
        heights[column] += 1;
    }

    public void UnmakeMove(int column, Player player)
    {
        if (player == Player.Red)
        {
            redCounts[column] -= 1;
            redRowCounts[heights[column] - 1] -= 1;
            zobristHash ^= zobristTable[2 * (6 * column + heights[column] - 1)];
        }
        else
        {
            yellowCounts[column] -= 1;
            yellowRowCounts[heights[column] - 1] -= 1;
            zobristHash ^= zobristTable[2 * (6 * column + heights[column] - 1) + 1];
        }
        board[6 * column + heights[column] - 1] = Player.None;
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
