using System;
using UnityEngine;

public static class NoiseUtils
{
    // Example Dirichlet sampler
    public static float[] SampleDirichlet(int dim, float alpha)
    {
        float[] gammas = new float[dim];
        float sum = 0f;
        for (int i = 0; i < dim; i++)
        {
            gammas[i] = SampleGamma(alpha, 1.0f);
            sum += gammas[i];
        }

        // Normalize
        for (int i = 0; i < dim; i++)
        {
            gammas[i] /= sum;
        }
        return gammas;
    }

    // Marsaglia-Tsang method for Gamma(shape=k, scale=theta),
    // but adapted to alpha >= 1. 
    // If alpha < 1, we do a common trick: gamma(alpha,1) = gamma(alpha+1,1) * U^(1/alpha), see below.
    public static float SampleGamma(float alpha, float scale)
    {
        if (alpha < 1)
        {
            // Trick: gamma(alpha, 1) = gamma(alpha+1, 1) * U^(1/alpha)
            float u = UnityEngine.Random.value; // uniform(0,1)
            return SampleGamma(alpha + 1f, 1f) * Mathf.Pow(u, 1f / alpha);
        }

        // Now alpha >= 1
        float d = alpha - 1f / 3f;
        float c = 1f / Mathf.Sqrt(9f * d);

        while (true)
        {
            float x = RandomNormal(0f, 1f); // standard normal
            float v = 1f + c * x;
            if (v <= 0) continue;

            v = v * v * v;
            float u = UnityEngine.Random.value;
            // Squeeze
            if (u < 1f - 0.0331f * x * x * x * x)
                return scale * d * v;
            if (Mathf.Log(u) < 0.5f * x * x + d * (1f - v + Mathf.Log(v)))
                return scale * d * v;
        }
    }

    // Box-Muller to sample from standard normal(0,1)
    public static float RandomNormal(float mean, float std)
    {
        float u1 = 1f - UnityEngine.Random.value; // avoid 0
        float u2 = 1f - UnityEngine.Random.value;
        float randStdNormal = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) *
                              Mathf.Sin(2.0f * Mathf.PI * u2);
        return mean + std * randStdNormal;
    }
}
