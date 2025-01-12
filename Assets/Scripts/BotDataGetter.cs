using UnityEngine;
using System.IO;
using UnityEngine.UI;

public class BotDataGetter : MonoBehaviour
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

    private float[] avgColumnWin;
    private float[] avgColumnLoss;
    private float[] avgRowWin;
    private float[] avgRowLoss;
    private float[,] randomValues;
    private float[] resultingScores;

    int currentGame = 0;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        boardA = new BoardTestA();
        boardB = new BoardTestB();

        avgColumnWin = new float[7];
        avgRowWin = new float[6];

        avgColumnLoss = new float[7];
        avgRowLoss = new float[6];

        for (int i = 0; i < 7; i++)
        {
            avgColumnWin[i] = 0;
            avgColumnLoss[i] = 0;
        }
        for (int i = 0; i < 6; i++)
        {
            avgRowWin[i] = 0;
            avgRowLoss[i] = 0;
        }

        randomValues = new float[numGames, 8];
        resultingScores = new float[numGames];
        for (int i = 0; i < numGames; i++) {
            resultingScores[i] = 0;
            for (int j = 0; j < 7; j++)
            {
                randomValues[i, j] = Random.Range(0f, 1f);
            }
            randomValues[i, 7] = (randomValues[i, 0] + randomValues[i, 1] + randomValues[i, 2] + randomValues[i, 3]) - (randomValues[i, 4] + randomValues[i, 5] + randomValues[i, 6]);
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (currentGame < numGames)
        {
            float[] valsA = new float[8];
            for (int i = 0; i < 8; i++)
            {
                valsA[i] = randomValues[currentGame, i];
            }
            for (int i = currentGame + 1; i < numGames; i++)
            {
                float[] valsB = new float[8];
                for (int j = 0; j < 8; j++)
                {
                    valsB[j] = randomValues[i, j];
                }

                BoardTestA.Player result1 = GetGameResult(depthA, depthB, valsA, valsB, true);

                // Result when A is red, B is yellow
                if (result1 == BoardTestA.Player.Red)
                {
                    resultingScores[currentGame] += 1;
                }
                else if (result1 == BoardTestA.Player.Yellow)
                {
                    resultingScores[i] += 1;
                }
                else
                {
                    resultingScores[i] += 0.5f;
                    resultingScores[currentGame] += 0.5f;
                }

                BoardTestA.Player result2 = GetGameResult(depthA, depthB, valsA, valsB, false);

                // Result when B is red, A is yellow
                if (result2 == BoardTestA.Player.Red)
                {
                    resultingScores[i] += 1;
                }
                else if (result2 == BoardTestA.Player.Yellow)
                {
                    resultingScores[currentGame] += 1;
                }
                else
                {
                    resultingScores[currentGame] += 0.5f;
                    resultingScores[i] += 0.5f;
                }
            }

            string outputLog = "SCORE " + resultingScores[currentGame].ToString() + " | Position values:";
            for (int i = 0; i < 7; i++)
            {
                outputLog += " ";
                outputLog += valsA[i].ToString();
            }
            Debug.Log(outputLog);

            currentGame += 1;
            bElo.text = currentGame.ToString() + "/" + numGames.ToString();
        }
    }

    BoardTestA.Player GetGameResult(int depthA, int depthB, float[] valsA, float[] valsB, bool aIsRed)
    {
        boardA.ResetBoard();
        boardB.ResetBoard();

        float[] trueValsA1 = new float[7];
        float[] trueValsA2 = new float[7];
        float[] trueValsB1 = new float[7];
        float[] trueValsB2 = new float[7];

        trueValsA1[3] = valsA[3];
        trueValsA2[3] = valsA[7];
        trueValsB1[3] = valsB[3];
        trueValsB2[3] = valsB[7];

        for (int i = 1; i < 4; i++)
        {
            trueValsA1[3-i] = valsA[3-i];
            trueValsA2[3-i] = valsA[7-i];
            trueValsB1[3-i] = valsB[3-i];
            trueValsB2[3-i] = valsB[7-i];

            trueValsA1[3+i] = valsA[3-i];
            trueValsA2[3+i] = valsA[7-i];
            trueValsB1[3+i] = valsB[3-i];
            trueValsB2[3+i] = valsB[7-i];
        }

        boardA.redPositionalVals = trueValsA1;
        boardA.yellowPositionVals = trueValsA2;
        boardB.redPositionalVals = trueValsA1;
        boardB.yellowPositionVals = trueValsA2;

        // BoardA as red, first
        if (aIsRed)
        {
            // A is red, B is yellow
            while (true)
            {
                boardA.tt.Clear();
                BoardTestA.MoveEval move = boardA.GetBestMove(depthA, Mathf.NegativeInfinity, Mathf.Infinity, true, -1);
                boardA.MakeMove(move.Move, BoardTestA.Player.Red);
                boardB.MakeMove(move.Move, BoardTestB.Player.Red);
                if (boardA.GetWinningPlayer() != BoardTestA.Player.None || boardA.IsFull())
                {
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
        }
        else
        {
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

    void WriteToFile(string fileName, string content)
    {
        string path = Application.persistentDataPath + "/" + fileName;

        // Write the content to the file
        File.WriteAllText(path, content);

        Debug.Log("File written to: " + path);
    }
}
