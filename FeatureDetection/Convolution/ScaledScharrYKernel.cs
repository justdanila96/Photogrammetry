namespace FeatureDetection.Convolution {
    internal class ScaledScharrYKernel : IKernel {
        public float[] Horizontal { get; }
        public float[] Vertical { get; }
        public ScaledScharrYKernel(int scale, bool norm = true) {
            int kSize = 3 + 2 * (scale - 1);

            Vertical = new float[kSize];
            Vertical[0] = -1f;
            Vertical[^1] = 1f;

            Horizontal = new float[kSize];
            const float w = 10f / 3f;
            Horizontal[0] = 1f;
            Horizontal[kSize / 2] = w;
            Horizontal[^1] = 1f;

            if (norm) {
                float N = 2f * scale * (w + 2f);
                for (int i = 0; i < Horizontal.Length; i++)
                    Horizontal[i] /= N;
            }
        }
    }
}
