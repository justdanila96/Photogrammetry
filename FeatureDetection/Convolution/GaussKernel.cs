namespace FeatureDetection.Convolution {
    internal class GaussKernel : IKernel {
        public float[] Horizontal { get; }
        public float[] Vertical { get; }
        public GaussKernel(int size, float sigma) {

            int kSize = (size == 0
                ? (int)MathF.Ceiling(2f * (1f + (sigma - .8f) / .3f))
                : size) | 1;

            float expScale = 1f / (2f * (sigma * sigma));

            float[] src =
                Enumerable
                .Range(-kSize / 2, kSize)
                .Select(i => MathF.Exp(-expScale * i * i))
                .ToArray();

            float sum = src.Sum();
            for (int i = 0; i < src.Length; i++)
                src[i] /= sum;

            Horizontal = Vertical = src;
        }
    }
}
