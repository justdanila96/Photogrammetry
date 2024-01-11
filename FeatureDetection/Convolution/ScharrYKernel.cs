namespace FeatureDetection.Convolution {
    internal class ScharrYKernel : IKernel {
        public float[] Horizontal { get; }
        public float[] Vertical { get; }
        public ScharrYKernel(bool norm = true) {
            Horizontal = [3f, 10f, 3f];
            if (norm) {
                for (int i = 0; i < Horizontal.Length; i++) {
                    Horizontal[i] /= 16f;
                }
            }
            Vertical = [-1f, 0f, 1f];
            if (norm) {
                for (int i = 0; i < Vertical.Length; i++) {
                    Vertical[i] /= 2f;
                }
            }
        }
    }
}
