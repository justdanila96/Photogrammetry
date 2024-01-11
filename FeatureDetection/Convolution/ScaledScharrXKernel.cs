namespace FeatureDetection.Convolution {
    internal class ScaledScharrXKernel : IKernel {
        public float[] Horizontal { get; }
        public float[] Vertical { get; }
        public ScaledScharrXKernel(int scale, bool norm = true) {
            int kSize = 3 + 2 * (scale - 1);

            Horizontal = new float[kSize];
            Horizontal[0] = -1f;
            Horizontal[^1] = 1f;

            const float w = 10f / 3f;
            Vertical = new float[kSize];
            Vertical[0] = 1f;
            Vertical[kSize / 2] = w;
            Vertical[^1] = 1f;

            if (norm) {
                float N = 2f * scale * (w + 2f);
                for (int i = 0; i < Vertical.Length; i++)
                    Vertical[i] /= N;
            }
        }
    }
}
