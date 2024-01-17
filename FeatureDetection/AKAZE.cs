using CommunityToolkit.HighPerformance;
using FeatureDetection.Convolution;
using ILGPU;
using ILGPU.Runtime;
using MathNet.Numerics.LinearAlgebra;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Collections;
using FBuffer2D = ILGPU.Runtime.MemoryBuffer2D<float, ILGPU.Stride2D.DenseX>;
using Index = (int i, int j);

namespace FeatureDetection {
    public partial class AKAZE(Accelerator accelerator, AKAZESettings settings) {

        private const float offset = 1.6f;
        private const float derivativeFactor = 1.5f;
        private const float minBorder = 14.1421f;
        private Layer[] layers = [];

        private IEnumerable<LayerMetadata> GetLayerMetadata() {

            for (int i = 0; i < settings.Levels; i++) {

                int ratio = 1 << i;
                for (int j = 0; j < settings.Sublevels; j++) {

                    float power = (float)j / settings.Sublevels + i;
                    float sigma = offset * MathF.Pow(2, power);
                    float sigmaSize = MathF.Round(sigma * derivativeFactor / ratio);

                    yield return new LayerMetadata {
                        Level = i,
                        Sublevel = j,
                        Ratio = ratio,
                        ESigma = sigma,
                        SigmaSize = (int)sigmaSize,
                        ETime = MathF.Pow(sigma, 2f) / 2f,
                        Border = 1 + (int)MathF.Round(minBorder * sigmaSize)
                    };
                }
            }
        }

        private void FillPyramid(Image<Rgb24> image) {

            LayerGPU[] layersGPU =
                layers.Select(layer =>
                    new LayerGPU(accelerator, new Index2D(layer.Width, layer.Height)))
                .ToArray();

            float kcontrast;
            var smooth = new GaussKernel(0, offset);
            using (FBuffer2D original = ToGrayscale(image)) {

                kcontrast = ComputeContrastFactor(original);
                SepConvolution(original, smooth, layersGPU[0].Lt);
            }

            layersGPU[0].Lt.CopyTo(layersGPU[0].LSmooth);

            smooth = new GaussKernel(0, 1f);
            var scharrX = new ScharrXKernel(false);
            var scharrY = new ScharrYKernel(false);

            for (int i = 1; i < layersGPU.Length; i++) {

                LayerGPU prevLayer = layersGPU[i - 1];
                LayerGPU curLayer = layersGPU[i];

                if (layers[i].Metadata.Level <= layers[i - 1].Metadata.Level)
                    prevLayer.Lt.CopyTo(curLayer.Lt);
                else {
                    Halfsize(prevLayer.Lt, curLayer.Lt);
                    kcontrast *= 0.75f;
                }

                SepConvolution(curLayer.Lt, smooth, curLayer.LSmooth);
                SepConvolution(curLayer.LSmooth, scharrX, curLayer.Lx);
                SepConvolution(curLayer.LSmooth, scharrY, curLayer.Ly);

                using FBuffer2D flow = PeronaMalikG2Diff(curLayer.Lx, curLayer.Ly, kcontrast);
                float delta = layers[i].Metadata.ETime - layers[i - 1].Metadata.ETime;
                float[] timings = ImgDiffusion.GetTimings(delta);
                ImageFEDCycle(curLayer.Lt, flow, timings);
            }

            for (int i = 0; i < layersGPU.Length; i++) {

                LayerGPU curLayer = layersGPU[i];

                Index2D extent = curLayer.Lt.IntExtent;
                int kernelSize = layers[i].Metadata.SigmaSize;
                var xKernel = new ScaledScharrXKernel(kernelSize);
                var yKernel = new ScaledScharrYKernel(kernelSize);

                using FBuffer2D lxx = accelerator.Allocate2DDenseX<float>(extent);
                using FBuffer2D lxy = accelerator.Allocate2DDenseX<float>(extent);
                using FBuffer2D lyy = accelerator.Allocate2DDenseX<float>(extent);

                SepConvolution(curLayer.LSmooth, xKernel, curLayer.Lx);
                SepConvolution(curLayer.LSmooth, yKernel, curLayer.Ly);
                SepConvolution(curLayer.Lx, xKernel, lxx);
                SepConvolution(curLayer.Lx, yKernel, lxy);
                SepConvolution(curLayer.Ly, yKernel, lyy);

                ComputeHessian(lxx, lxy, lyy, MathF.Pow(kernelSize, 4f), curLayer.Ldet);
            }

            _ = Parallel.For(0, layers.Length, i => {

                LayerGPU layerGPU = layersGPU[i];
                float[] lt = layerGPU.Lt.View.To1DView().GetAsArray1D();
                float[] lx = layerGPU.Lx.View.To1DView().GetAsArray1D();
                float[] ly = layerGPU.Ly.View.To1DView().GetAsArray1D();
                float[] ldet = layerGPU.Ldet.View.To1DView().GetAsArray1D();

                layers[i] = layers[i] with { Lt = lt, Lx = lx, Ly = ly, Ldet = ldet };
                layerGPU.Dispose();
            });
        }

        private static bool FindNeighbor(Index inPos, HashSet<Index> points, int R, out Index point) {

            point = (0, 0);
            int R2 = R * R;

            for (int i = inPos.i - R; i < inPos.i + R; i++) {
                for (int j = inPos.j - R; j < inPos.j + R; j++) {

                    var index = (i, j);

                    if (!points.Contains(index))
                        continue;

                    int dx = j - inPos.j;
                    int dy = i - inPos.i;

                    if (dx * dx + dy * dy <= R2) {
                        point = index;
                        return true;
                    }
                }
            }

            return false;
        }

        private static HashSet<Index> FindKeypoints(Layer step, float threshold) {

            var result = new HashSet<Index>();
            ReadOnlySpan2D<float> Ldet = step.LdetAsMatrix();

            for (int i = step.Metadata.Border; i < step.Height - step.Metadata.Border; i++) {
                for (int j = step.Metadata.Border; j < step.Width - step.Metadata.Border; j++) {

                    float val = Ldet[i, j];

                    if (val <= threshold
                     || val <= Ldet[i - 1, j - 1]
                     || val <= Ldet[i - 1, j]
                     || val <= Ldet[i - 1, j + 1]
                     || val <= Ldet[i, j - 1]
                     || val <= Ldet[i, j + 1]
                     || val <= Ldet[i + 1, j - 1]
                     || val <= Ldet[i + 1, j]
                     || val <= Ldet[i + 1, j + 1]) {
                        continue;
                    }

                    Index index = (i, j);

                    if (FindNeighbor(index, result, step.Metadata.SigmaSize, out Index point)) {

                        (int pi, int pj) = point;

                        if (val > Ldet[pi, pj]) {
                            result.Remove(point);
                        }
                        else {
                            continue;
                        }
                    }

                    result.Add(index);
                }
            }

            return result;
        }

        private HashSet<Index>[] FindScaleSpaceExtrema() {

            var kptsByLayers = new HashSet<Index>[layers.Length];
            Parallel.For(0, layers.Length, i => {
                kptsByLayers[i] = FindKeypoints(layers[i], settings.Threshold);
            });

            for (int k = 1; k < layers.Length; k++) {

                HashSet<Index> kpts = kptsByLayers[k];
                if (kpts.Count == 0)
                    continue;

                HashSet<Index> kptsPrev = kptsByLayers[k - 1];
                if (kptsPrev.Count == 0)
                    continue;

                ReadOnlySpan2D<float> Ldet = layers[k].LdetAsMatrix();
                ReadOnlySpan2D<float> LdetPrev = layers[k - 1].LdetAsMatrix();
                int ratio = layers[k].Metadata.Ratio / layers[k - 1].Metadata.Ratio;
                int radius = layers[k].Metadata.SigmaSize * ratio;

                foreach ((int i, int j) in kpts) {

                    Index scaledIndex = (i * ratio, j * ratio);

                    if (FindNeighbor(scaledIndex, kptsPrev, radius, out Index neighbor)
                        && Ldet[i, j] > LdetPrev[neighbor.i, neighbor.j]) {

                        kptsPrev.Remove(neighbor);
                    }
                }
            }

            for (int k = layers.Length - 2; k >= 0; k--) {

                HashSet<Index> kpts = kptsByLayers[k];
                if (kpts.Count == 0)
                    continue;

                HashSet<Index> kptsNext = kptsByLayers[k + 1];
                if (kptsNext.Count == 0)
                    continue;

                ReadOnlySpan2D<float> Ldet = layers[k].LdetAsMatrix();
                ReadOnlySpan2D<float> LdetNext = layers[k + 1].LdetAsMatrix();

                int ratio = layers[k + 1].Metadata.Ratio / layers[k].Metadata.Ratio;
                int radius = layers[k + 1].Metadata.SigmaSize;

                foreach ((int i, int j) in kpts) {

                    Index scaledIndex = (i / ratio, j / ratio);

                    if (FindNeighbor(scaledIndex, kptsNext, radius, out Index neighbor)
                        && Ldet[i, j] > LdetNext[neighbor.i, neighbor.j]) {

                        kptsNext.Remove(neighbor);
                    }
                }
            }

            return kptsByLayers;
        }

        private static Vector<float> KeypointCorrection(int i, int j, ReadOnlySpan2D<float> ldet) {

            float Dx = (ldet[i, j + 1] - ldet[i, j - 1]) / 2f;
            float Dy = (ldet[i + 1, j] - ldet[i - 1, j]) / 2f;

            float Dxx = ldet[i, j + 1] + ldet[i, j - 1] - 2f * ldet[i, j];
            float Dyy = ldet[i + 1, j] + ldet[i - 1, j] - 2f * ldet[i, j];
            float Dxy = (ldet[i + 1, j + 1] + ldet[i - 1, j - 1] - ldet[i - 1, j + 1] - ldet[i + 1, j - 1]) / 4f;

            var A = Matrix<float>.Build.DenseOfRowMajor(2, 2, [Dxx, Dxy, Dxy, Dyy]);
            var B = Vector<float>.Build.DenseOfArray([-Dx, -Dy]);
            return A.Solve(B);
        }

        private List<Keypoint> DetectKeypoints(HashSet<Index>[] kptsByLayers) {

            object locker = new();
            List<Keypoint> result = [];

            Parallel.For(0, layers.Length,

                () => new List<Keypoint>(),

                (k, _, list) => {

                    HashSet<Index> kpts = kptsByLayers[k];
                    if (kpts.Count == 0)
                        return list;

                    Layer step = layers[k];
                    int ratio = step.Metadata.Ratio;
                    int level = step.Metadata.Level;
                    float sz = 2f * step.Metadata.ESigma * derivativeFactor;
                    int sz0 = (int)MathF.Round(.5f * sz / ratio);

                    ReadOnlySpan2D<float> ldet = step.LdetAsMatrix();
                    ReadOnlySpan2D<float> lx = step.LxAsMatrix();
                    ReadOnlySpan2D<float> ly = step.LyAsMatrix();

                    foreach ((int i, int j) in kpts) {

                        var corr = KeypointCorrection(i, j, ldet);
                        if (corr.Any(k => MathF.Abs(k) > 1f))
                            continue;

                        float x = j * ratio + corr[0] * ratio + (ratio - 1f) / 2f;
                        float y = i * ratio + corr[1] * ratio + (ratio - 1f) / 2f;

                        int x0 = (int)MathF.Round(x / ratio);
                        int y0 = (int)MathF.Round(y / ratio);
                        float angle = KeypointProcessing.ComputeAngle(lx, ly, x0, y0, sz0);

                        list.Add(new Keypoint(x, y, sz, angle, ldet[i, j], level, k));
                    }

                    return list;
                },

                list => {
                    if (list.Count != 0) {
                        lock (locker) {
                            result.AddRange(list);
                        }
                    }
                });

            return result;
        }

        public void GetKeypoints(Stream imgStream, out List<Keypoint> kpts, out List<BitArray> descr) {

            using (var img = Image.Load<Rgb24>(imgStream)) {

                layers = GetLayerMetadata()
                    .Select(metadata => {
                        int width = img.Width / metadata.Ratio;
                        int height = img.Height / metadata.Ratio;
                        return new Layer(metadata, width, height);
                    })
                    .ToArray();

                FillPyramid(img);
            }

            HashSet<Index>[] kptsByLayers = FindScaleSpaceExtrema();
            kpts = DetectKeypoints(kptsByLayers);

            descr = [];
            foreach (var kpt in kpts) {
                descr.Add(MLDB.ComputeDescriptor(layers[kpt.ClassId], kpt));
            }
        }

        public void TestPyramid(Stream imgStream, string outputFolder) {

            using (var img = Image.Load<Rgb24>(imgStream)) {

                layers = GetLayerMetadata()
                    .Select(metadata => {
                        int width = img.Width / metadata.Ratio;
                        int height = img.Height / metadata.Ratio;
                        return new Layer(metadata, width, height);
                    })
                    .ToArray();

                FillPyramid(img);
            }

            for (int i = 0; i < layers.Length; i++) {

                TestLoadImage(i, layers[i], outputFolder);
            }
        }
    }
}
