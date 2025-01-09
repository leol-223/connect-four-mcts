using UnityEngine;
using UnityEngine.UI;

public class BotTuner : MonoBehaviour
{
    public int numGames;
    public int depthA;
    public int depthB;
    public TMPro.TMP_Text bWins;
    public TMPro.TMP_Text bDraws;
    public TMPro.TMP_Text bLosses;
    public TMPro.TMP_Text bElo;

    private BoardTestA boardA;
    private BoardTestB boardB;

    private int numWins;
    private int numDraws;
    private int numLosses;
    private int numGamesPlayed;

    private float[] avgColumnWin;
    private float[] avgColumnLoss;
    private float[] avgRowWin;
    private float[] avgRowLoss;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        boardA = new BoardTestA();
        boardB = new BoardTestB();

        avgColumnWin = new float[7];
        avgRowWin = new float[6];

        avgColumnLoss = new float[7];
        avgRowLoss = new float[6];

        for (int i = 0; i < 7; i++) {
            avgColumnWin[i] = 0;
            avgColumnLoss[i] = 0;
        }
        for (int i = 0; i < 6; i++) {
            avgRowWin[i] = 0;
            avgRowLoss[i] = 0;
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (numGamesPlayed < numGames)
        {
            BoardTestA.Player result1 = GetGameResult(depthA, depthB, true);
            if (result1 != BoardTestA.Player.None)
            {
                for (int i = 0; i < boardA.board.Length; i++)
                {
                    BoardTestA.Player token = boardA.board[i];
                    if (token == BoardTestA.Player.None)
                    {
                        continue;
                    }
                    int column = i / 6;
                    int row = i % 6;
                    if (token == result1)
                    {
                        avgColumnWin[column] += 0f;
                        avgRowWin[row] += 0f;
                    }
                    else
                    {
                        avgColumnLoss[column] += 0f;
                        avgRowLoss[row] += 0f;
                    }
                }
            }

            // Result when A is red, B is yellow
            if (result1 == BoardTestA.Player.Red)
            {
                numLosses += 1;
            }
            else if (result1 == BoardTestA.Player.Yellow)
            {
                numWins += 1;
            }
            else {
                numDraws += 1;
            }
            numGamesPlayed += 1;

            BoardTestA.Player result2 = GetGameResult(depthA, depthB, false);
            if (result2 != BoardTestA.Player.None)
            {
                for (int i = 0; i < boardA.board.Length; i++)
                {
                    BoardTestA.Player token = boardA.board[i];
                    if (token == BoardTestA.Player.None)
                    {
                        continue;
                    }
                    int column = i / 6;
                    int row = i % 6;
                    if (token == result2)
                    {
                        avgColumnWin[column] += 1f;
                        avgRowWin[row] += 1f;
                    }
                    else
                    {
                        avgColumnLoss[column] += 1f;
                        avgRowLoss[row] += 1f;
                    }
                }
            }

            // Result when B is red, A is yellow
            if (result2 == BoardTestA.Player.Red)
            {
                numWins += 1;
            }
            else if (result2 == BoardTestA.Player.Yellow)
            {
                numLosses += 1;
            }
            else
            {
                numDraws += 1;
            }

            numGamesPlayed += 1;

            
            string rowString = "Rows: ";
            for (int i = 0; i < 6; i++)
            {
                rowString += ((avgRowWin[i] - avgRowLoss[i]) / (float)numGamesPlayed).ToString();
                rowString += ", ";
            }
            string columnString = "Columns: ";
            for (int i = 0; i < 7; i++)
            {
                columnString += ((avgColumnWin[i] - avgColumnLoss[i]) / (float)numGamesPlayed).ToString();
                columnString += ", ";
            }
            Debug.Log(rowString);
            Debug.Log(columnString);
            
        }
        float N = numWins + numDraws + numLosses;
        if (N > 0)
        {
            float w = (numWins + numDraws / 2) / N;
            if (w < 1 && w > 0)
            {
                float eloDif = -400 * Mathf.Log10(1 / w - 1);
                float se = 400 / (Mathf.Sqrt(N) * Mathf.Log(10));
                string variation = Mathf.RoundToInt(1.96f * se).ToString();
                if (eloDif > 0)
                {
                    bElo.text = "Elo change: +" + Mathf.RoundToInt(eloDif).ToString() + " (Variation: " + variation + ")";
                } else
                {
                    bElo.text = "Elo change: " + Mathf.RoundToInt(eloDif).ToString() + " (Variation: " + variation + ")";
                }
            }
        }
        bWins.text = numWins.ToString() + " wins";
        bDraws.text = numDraws.ToString() + " draws";
        bLosses.text = numLosses.ToString() + " losses";
    }

    BoardTestA.Player GetGameResult(int depthA, int depthB, bool aIsRed) {
        boardA.ResetBoard();
        boardB.ResetBoard();

        // BoardA as red, first
        if (aIsRed) {
            // A is red, B is yellow
            while (true)
            {
                boardA.tt.Clear();
                BoardTestA.MoveEval move = boardA.GetBestMove(depthA, Mathf.NegativeInfinity, Mathf.Infinity, true, -1);
                boardA.MakeMove(move.Move, BoardTestA.Player.Red);
                boardB.MakeMove(move.Move, BoardTestB.Player.Red);
                if (boardA.GetWinningPlayer() != BoardTestA.Player.None || boardA.IsFull()) {
                    break;
                }
                boardB.tt.Clear();
                BoardTestB.MoveEval move2 = boardB.GetBestMove(depthB, Mathf.NegativeInfinity, Mathf.Infinity, false, -1);
                boardA.MakeMove(move2.Move, BoardTestA.Player.Yellow);
                boardB.MakeMove(move2.Move, BoardTestB.Player.Yellow);
                if (boardA.GetWinningPlayer() != BoardTestA.Player.None || boardA.IsFull())
                {
                    break;
                }
            }
        } else {
            // A is yellow, B is red
            while (true)
            {
                boardB.tt.Clear();
                BoardTestB.MoveEval move = boardB.GetBestMove(depthB, Mathf.NegativeInfinity, Mathf.Infinity, true, -1);
                boardA.MakeMove(move.Move, BoardTestA.Player.Red);
                boardB.MakeMove(move.Move, BoardTestB.Player.Red);
                if (boardA.GetWinningPlayer() != BoardTestA.Player.None || boardA.IsFull())
                {
                    break;
                }

                boardA.tt.Clear();
                BoardTestA.MoveEval move2 = boardA.GetBestMove(depthA, Mathf.NegativeInfinity, Mathf.Infinity, false, -1);
                boardA.MakeMove(move2.Move, BoardTestA.Player.Yellow);
                boardB.MakeMove(move2.Move, BoardTestB.Player.Yellow);
                if (boardA.GetWinningPlayer() != BoardTestA.Player.None || boardA.IsFull())
                {
                    break;
                }
            }
        }
        return boardA.GetWinningPlayer();
    }
}
