using CommunityToolkit.HighPerformance;
using System.Collections.Immutable;
using System.Numerics;

namespace FeatureDetection {
    internal static class KeypointProcessing {

        private static readonly float[,] Gauss25 = new float[,] {

            { 0.02546481f, 0.02350698f, 0.01849125f, 0.01239505f, 0.00708017f, 0.00344629f, 0.00142946f },
            { 0.02350698f, 0.02169968f, 0.01706957f, 0.01144208f, 0.00653582f, 0.00318132f, 0.00131956f },
            { 0.01849125f, 0.01706957f, 0.01342740f, 0.00900066f, 0.00514126f, 0.00250252f, 0.00103800f },
            { 0.01239505f, 0.01144208f, 0.00900066f, 0.00603332f, 0.00344629f, 0.00167749f, 0.00069579f },
            { 0.00708017f, 0.00653582f, 0.00514126f, 0.00344629f, 0.00196855f, 0.00095820f, 0.00039744f },
            { 0.00344629f, 0.00318132f, 0.00250252f, 0.00167749f, 0.00095820f, 0.00046640f, 0.00019346f },
            { 0.00142946f, 0.00131956f, 0.00103800f, 0.00069579f, 0.00039744f, 0.00019346f, 0.00008024f }
        };

        private readonly record struct GTabItem(int X, int Y, float Weight);

        private static IEnumerable<GTabItem> GenGTab() {
            for (int x = -6; x <= 6; x++) {
                for (int y = -6; y <= 6; y++) {
                    if (x * x + y * y < 36)
                        yield return new(x, y, Gauss25[Math.Abs(x), Math.Abs(y)]);
                }
            }
        }

        private static readonly ImmutableArray<GTabItem> GTab = GenGTab().ToImmutableArray();      

        private static void QuantizedCountingSort(float[] a, int n, float quantum, int nkeys, out int[] idx, out int[] cum) {

            int clamp(int k) => k < 0 || k >= nkeys ? 0 : k;
            int getIndex(float val) => clamp((int)(val / quantum));

            cum = new int[nkeys + 1];
            for (int i = 0; i < n; i++) {
                int b = getIndex(a[i]);
                cum[b]++;
            }

            for (int i = 1; i <= nkeys; i++) {
                cum[i] += cum[i - 1];
            }

            idx = new int[n];
            for (int i = 0; i < n; i++) {
                int b = getIndex(a[i]);
                idx[--cum[b]] = i;
            }
        }

        public static float ComputeAngle(ReadOnlySpan2D<float> Lx, ReadOnlySpan2D<float> Ly, int x0, int y0, int scale) {

            const int angSize = 109;
            int N = GTab.Length;

            var res = new Vector2[N];
            for (int i = 0; i < N; i++) {

                (int xidx, int yidx, float w) = GTab[i];
                int y = y0 + yidx * scale;
                int x = x0 + xidx * scale;
                res[i] = new Vector2(w * Lx[y, x], w * Ly[y, x]);
            }

            var angles = res.Select(v => MathF.Atan2(v.Y, v.X)).ToArray();

            const int slices = 42;
            float angStep = 2f * MathF.PI / slices;
            QuantizedCountingSort(angles, angSize, angStep, slices, out int[] sortedIdx, out int[] slice);

            Vector2 VectorSum(int from, int to, Vector2 init) =>
                Enumerable.Range(from, to - from).Aggregate(init, (sum, i) =>
                    sum + res[sortedIdx[i]]);

            float VectorNorm(Vector2 v) => v.X * v.X + v.Y * v.Y;

            const int win = 7;
            var max = VectorSum(slice[0], slice[win], Vector2.Zero);
            float maxNorm = VectorNorm(max);

            for (int sn = 1; sn <= slices - win; sn++) {

                if (slice[sn] == slice[sn - 1] && slice[sn + win] == slice[sn + win - 1]) {
                    continue;
                }
                var sum = VectorSum(slice[sn], slice[sn + win], Vector2.Zero);
                float norm = VectorNorm(sum);

                if (norm > maxNorm) {
                    maxNorm = norm;
                    max += sum;
                }
            }

            for (int sn = slices - win + 1; sn < slices; sn++) {

                int remain = sn + win - slices;
                if (slice[sn] == slice[sn - 1] && slice[remain] == slice[remain - 1]) {
                    continue;
                }
                Vector2 sum = VectorSum(slice[sn], slice[slices], Vector2.Zero);
                sum = VectorSum(slice[0], slice[remain], sum);
                float norm = VectorNorm(sum);

                if (norm > maxNorm) {
                    maxNorm = norm;
                    max += sum;
                }
            }

            return MathF.Atan2(max.Y, max.X);
        }
    }
}
