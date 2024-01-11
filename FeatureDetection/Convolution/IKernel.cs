namespace FeatureDetection.Convolution {
    internal interface IKernel {
        float[] Horizontal { get; }
        float[] Vertical { get; }
    }
}
