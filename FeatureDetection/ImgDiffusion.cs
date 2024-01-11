namespace FeatureDetection {
    internal static class ImgDiffusion {
        private static bool IsPrime(int n) {

            if (n <= 1)
                return false;
            else if (n == 2)
                return true;
            else if (n % 2 == 0)
                return false;
            else {
                int bound = (int)MathF.Floor(MathF.Sqrt(n));
                for (int i = 3; i <= bound; i += 2) {
                    if (n % i == 0)
                        return false;
                }
                return true;
            }
        }

        private static int NextPrime(int i) => IsPrime(i) ? i : NextPrime(i + 1);

        public static float[] GetTimings(float T, float Tmax = .25f) {

            int n = (int)MathF.Ceiling(MathF.Sqrt(3f * T / Tmax + .25f) - .5f - 1e-8f);
            float scale = 3f * T / (Tmax * (n * n + n));
            float cosFact = 1f / (4 * n + 2f);
            float gloFact = scale * Tmax / 2f;

            var tmp = new float[n];
            for (int j = 0; j < n; ++j) {
                float cosj = MathF.Cos(MathF.PI * (2 * j + 1) * cosFact);
                tmp[j] = gloFact / (cosj * cosj);
            }

            int kappa = n / 2;
            int p = NextPrime(n + 1);
            var tau = new float[n];

            for (int i = 0, k = 0; i < n; ++i, ++k) {

                int index;
                while ((index = (k + 1) * kappa % p - 1) >= n) {
                    ++k;
                }

                tau[i] = tmp[index];
            }

            return tau;
        }
    }
}
