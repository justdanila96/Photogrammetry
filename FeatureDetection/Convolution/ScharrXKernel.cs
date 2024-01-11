namespace FeatureDetection.Convolution {
    internal class ScharrXKernel : IKernel {
        public float[] Horizontal { get; }
        public float[] Vertical { get; }
        public ScharrXKernel(bool norm = true) {
            Horizontal = [-1f, 0f, 1f];
            if (norm) {
                for (int i = 0; i < Horizontal.Length; i++) {
                    Horizontal[i] /= 2f;
                }
            }
            Vertical = [3f, 10f, 3f];
            if (norm) {
                for (int i = 0; i < Vertical.Length; i++) {
                    Vertical[i] /= 16f;
                }
            }
        }
    }
}
