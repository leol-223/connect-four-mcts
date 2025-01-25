using System;
using System.Threading;

public static class NoiseUtils
{
    private static readonly ThreadLocal<Random> random = 
        new ThreadLocal<Random>(() => new Random(Interlocked.Increment(ref randomSeed)));
    private static int randomSeed = Environment.TickCount ^ Guid.NewGuid().GetHashCode();

    public static float[] SampleDirichlet(int dim, float alpha)
    {
        float[] gammas = new float[dim];
        float sum = 0f;
        for (int i = 0; i < dim; i++)
        {
            gammas[i] = SampleGamma(alpha, 1.0f);
            sum += gammas[i];
        }

        // Protect against division by zero
        if (sum <= float.Epsilon)
        {
            // Fallback to uniform distribution
            float uniformValue = 1.0f / dim;
            for (int i = 0; i < dim; i++)
            {
                gammas[i] = uniformValue;
            }
            return gammas;
        }

        // Normalize
        for (int i = 0; i < dim; i++)
        {
            gammas[i] /= sum;
        }
        return gammas;
    }

    public static float SampleGamma(float alpha, float scale)
    {
        // Protect against invalid inputs
        if (float.IsNaN(alpha) || float.IsNaN(scale) || alpha <= 0 || scale <= 0)
        {
            return 0.0f;
        }

        if (alpha < 1)
        {
            float u = (float)random.Value.NextDouble();
            return SampleGamma(alpha + 1f, 1f) * (float)Math.Pow(u, 1f / alpha);
        }

        float d = alpha - 1f / 3f;
        float c = 1f / (float)Math.Sqrt(9f * d);

        while (true)
        {
            float x = RandomNormal(0f, 1f);
            float v = 1f + c * x;
            if (v <= 0) continue;

            v = v * v * v;
            float u = (float)random.Value.NextDouble();
            
            if (u < 1f - 0.0331f * x * x * x * x)
                return scale * d * v;
            if (Math.Log(u) < 0.5f * x * x + d * (1f - v + Math.Log(v)))
                return scale * d * v;
        }
    }

    public static float RandomNormal(float mean, float std)
    {
        float u1 = 1f - (float)random.Value.NextDouble();
        float u2 = 1f - (float)random.Value.NextDouble();
        float randStdNormal = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) *
                              Math.Sin(2.0 * Math.PI * u2));
        return mean + std * randStdNormal;
    }
}
