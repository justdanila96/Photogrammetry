using CommunityToolkit.HighPerformance;
using ILGPU;
using ILGPU.Runtime;
using FBuffer2D = ILGPU.Runtime.MemoryBuffer2D<float, ILGPU.Stride2D.DenseX>;

namespace FeatureDetection {
    public readonly record struct AKAZESettings(int Levels, int Sublevels, float Threshold) {
        public AKAZESettings() : this(4, 4, 0.0005f) { }
    }

    internal readonly record struct LayerMetadata(int Level, int Sublevel, int Ratio, float ETime, float ESigma, int SigmaSize, int Border);

    internal readonly record struct Layer(LayerMetadata Metadata, int Width, int Height, float[]? Lt, float[]? Lx, float[]? Ly, float[]? Ldet) {
        public Layer(LayerMetadata metadata, int width, int height) : this(metadata, width, height, null, null, null, null) { }
    }

    internal static class LayerExtensions {
        public static ReadOnlySpan2D<float> LtAsMatrix(this Layer layer) =>
            layer.Lt != null
            ? new(layer.Lt, layer.Height, layer.Width)
            : throw new NullReferenceException("Lt is empty");
        public static ReadOnlySpan2D<float> LxAsMatrix(this Layer layer) =>
            layer.Lx != null
            ? new(layer.Lx, layer.Height, layer.Width)
            : throw new NullReferenceException("Lx is empty");
        public static ReadOnlySpan2D<float> LyAsMatrix(this Layer layer) =>
            layer.Ly != null
            ? new(layer.Ly, layer.Height, layer.Width)
            : throw new NullReferenceException("Ly is empty");
        public static ReadOnlySpan2D<float> LdetAsMatrix(this Layer layer) =>
            layer.Ldet != null
            ? new(layer.Ldet, layer.Height, layer.Width)
            : throw new NullReferenceException("Ldet is empty");
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