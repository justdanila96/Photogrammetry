using CommunityToolkit.HighPerformance;
using ILGPU;
using ILGPU.Runtime;
using FBuffer2D = ILGPU.Runtime.MemoryBuffer2D<float, ILGPU.Stride2D.DenseX>;

namespace FeatureDetection {
    public readonly record struct AKAZESettings(int Levels, int Sublevels, float Threshold) {
        public AKAZESettings() : this(4, 4, 0.0005f) { }
    }

    internal readonly record struct LayerMetadata(int Level, int Sublevel, int Ratio, float ETime, float ESigma, int SigmaSize, int Border);

    internal readonly record struct Layer(LayerMetadata Metadata, int Width, int Height, float[] Lt, float[] Lx, float[] Ly, float[] Ldet);

    internal static class LayerExtensions {
        public static ReadOnlySpan2D<float> LtAsMatrix(this Layer layer) => new(layer.Lt, layer.Height, layer.Width);
        public static ReadOnlySpan2D<float> LxAsMatrix(this Layer layer) => new(layer.Lx, layer.Height, layer.Width);
        public static ReadOnlySpan2D<float> LyAsMatrix(this Layer layer) => new(layer.Ly, layer.Height, layer.Width);
        public static ReadOnlySpan2D<float> LdetAsMatrix(this Layer layer) => new(layer.Ldet, layer.Height, layer.Width);
    }

    internal readonly record struct LayerGPU(FBuffer2D Lt, FBuffer2D LSmooth, FBuffer2D Lx, FBuffer2D Ly, FBuffer2D Ldet) {
        public LayerGPU(Accelerator accelerator, Index2D extent) : this(
            Lt: accelerator.Allocate2DDenseX<float>(extent),
            LSmooth: accelerator.Allocate2DDenseX<float>(extent),
            Lx: accelerator.Allocate2DDenseX<float>(extent),
            Ly: accelerator.Allocate2DDenseX<float>(extent),
            Ldet: accelerator.Allocate2DDenseX<float>(extent)) { }

        public void Dispose() {
            Lt.Dispose();
            LSmooth.Dispose();
            Lx.Dispose();
            Ly.Dispose();
            Ldet.Dispose();
        }
    }

    public readonly record struct Keypoint(float X, float Y, float Size, float Angle, float Response, int Octave, int ClassId);
}