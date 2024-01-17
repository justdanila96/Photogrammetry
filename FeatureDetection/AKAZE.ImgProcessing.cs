using FeatureDetection.Convolution;
using ILGPU;
using ILGPU.Runtime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using ByteArray = ILGPU.Runtime.ArrayView1D<byte, ILGPU.Stride1D.Dense>;
using ByteBuffer = ILGPU.Runtime.MemoryBuffer1D<byte, ILGPU.Stride1D.Dense>;
using FArray = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;
using FBuffer2D = ILGPU.Runtime.MemoryBuffer2D<float, ILGPU.Stride2D.DenseX>;
using FMatrix = ILGPU.Runtime.ArrayView2D<float, ILGPU.Stride2D.DenseX>;

namespace FeatureDetection {
    public partial class AKAZE {

        private readonly Action<Index2D, ByteArray, int, int, FMatrix> grayscaleGPU = accelerator.LoadAutoGroupedStreamKernel<Index2D, ByteArray, int, int, FMatrix>(GPU.ToGrayscale);
        private readonly Action<Index2D, FMatrix, FMatrix, FMatrix> gradientGPU = accelerator.LoadAutoGroupedStreamKernel<Index2D, FMatrix, FMatrix, FMatrix>(GPU.ComputeGradient);
        private readonly Action<Index2D, FMatrix, FMatrix> halfsizeGPU = accelerator.LoadAutoGroupedStreamKernel<Index2D, FMatrix, FMatrix>(GPU.Halfsize);
        private readonly Action<Index2D, FMatrix, FMatrix, FMatrix, float, FMatrix> hessianGPU = accelerator.LoadAutoGroupedStreamKernel<Index2D, FMatrix, FMatrix, FMatrix, float, FMatrix>(GPU.ComputeHessian);
        private readonly Action<Index2D, FMatrix, FArray, FMatrix> hConvGPU = accelerator.LoadAutoGroupedStreamKernel<Index2D, FMatrix, FArray, FMatrix>(GPU.HorizontalConvolution);
        private readonly Action<Index2D, FMatrix, FArray, FMatrix> vConvGPU = accelerator.LoadAutoGroupedStreamKernel<Index2D, FMatrix, FArray, FMatrix>(GPU.VerticalConvolution);
        private readonly Action<Index2D, FMatrix, FMatrix, float, FMatrix> diffusionGPU = accelerator.LoadAutoGroupedStreamKernel<Index2D, FMatrix, FMatrix, float, FMatrix>(GPU.PeronaMalikG2Diff);
        private readonly Action<Index2D, FMatrix, FMatrix, float, FMatrix> fedGPU = accelerator.LoadAutoGroupedStreamKernel<Index2D, FMatrix, FMatrix, float, FMatrix>(GPU.CentralFED);
        private readonly Action<Index2D, FMatrix, FMatrix> updSrcGPU = accelerator.LoadAutoGroupedStreamKernel<Index2D, FMatrix, FMatrix>(GPU.UpdateSrc);

        private FBuffer2D ToGrayscale(Image<Rgb24> src) {

            int bpp = src.PixelType.BitsPerPixel / 8;
            int stride = src.Width * bpp;
            var srcData = new byte[stride * src.Height];
            src.CopyPixelDataTo(srcData);
            using ByteBuffer srcBuffer = accelerator.Allocate1D(srcData);
            var extent = new Index2D(src.Width, src.Height);
            FBuffer2D output = accelerator.Allocate2DDenseX<float>(extent);
            grayscaleGPU(extent, srcBuffer.View, bpp, stride, output);
            return output;
        }

        private FBuffer2D ComputeGradient(FBuffer2D lx, FBuffer2D ly, int pad = 1) {

            var extent = new Index2D(lx.IntExtent.X - 2 * pad, lx.IntExtent.Y - 2 * pad);
            var offset = new Index2D(pad, pad);
            FMatrix lxSubview = lx.View.SubView(offset, extent);
            FMatrix lySubview = ly.View.SubView(offset, extent);
            FBuffer2D grad = accelerator.Allocate2DDenseX<float>(extent);
            gradientGPU(extent, lxSubview, lySubview, grad.View);
            return grad;
        }

        private void SepConvolution(FBuffer2D input, IKernel kernel, FBuffer2D output) {

            using var tmp = accelerator.Allocate2DDenseX<float>(input.Extent);

            Index2D hExtent = input.IntExtent + new Index2D(kernel.Horizontal.Length - 1, 0);
            using var hKernel = accelerator.Allocate1D(kernel.Horizontal);
            hConvGPU(hExtent, input.View, hKernel.View, tmp.View);

            Index2D vExtent = input.IntExtent + new Index2D(0, kernel.Vertical.Length - 1);
            using var vKernel = accelerator.Allocate1D(kernel.Vertical);
            vConvGPU(vExtent, tmp.View, vKernel.View, output.View);
        }

        private void Halfsize(FBuffer2D input, FBuffer2D output) =>
            halfsizeGPU(output.IntExtent, input.View, output.View);

        private void ComputeHessian(FBuffer2D lxx, FBuffer2D lxy, FBuffer2D lyy, float sigma, FBuffer2D output) =>
            hessianGPU(lxx.IntExtent, lxx.View, lxy.View, lyy.View, sigma, output.View);

        private FBuffer2D PeronaMalikG2Diff(FBuffer2D lx, FBuffer2D ly, float kcontrast) {

            FBuffer2D diff = accelerator.Allocate2DDenseX<float>(lx.Extent);
            diffusionGPU(lx.IntExtent, lx.View, ly.View, kcontrast * kcontrast, diff.View);
            return diff;
        }

        private void ImageFEDCycle(FBuffer2D src, FBuffer2D diff, float[] tau) {

            using var tmp = accelerator.Allocate2DDenseX<float>(src.Extent);
            foreach (float t in tau) {

                fedGPU(src.IntExtent, src.View, diff.View, t / 2f, tmp.View);
                updSrcGPU(src.IntExtent, src.View, tmp.View);
            }
        }

        private float ComputeContrastFactor(FBuffer2D src) {

            const int nbins = 300;
            const float percentile = .7f;

            var smooth = new GaussKernel(0, 1f);
            using FBuffer2D smoothed = accelerator.Allocate2DDenseX<float>(src.Extent);
            SepConvolution(src, smooth, smoothed);

            var scharrX = new ScharrXKernel(false);
            using FBuffer2D lx = accelerator.Allocate2DDenseX<float>(src.Extent);
            SepConvolution(smoothed, scharrX, lx);

            var scharrY = new ScharrYKernel(false);
            using FBuffer2D ly = accelerator.Allocate2DDenseX<float>(src.Extent);
            SepConvolution(smoothed, scharrY, ly);

            using FBuffer2D gradBuffer = ComputeGradient(lx, ly);
            float[] grad = gradBuffer.View.To1DView().GetAsArray1D();
            float gradMax = grad.Max();

            var locker = new object();
            var histogram = new int[nbins];

            Parallel.ForEach(
                grad,

                () => new int[nbins],

                (value, _, local) => {
                    if (value > 0f) {
                        int binId = Math.Clamp((int)(nbins * (value / gradMax)), 0, nbins - 1);
                        local[binId]++;
                    }
                    return local;
                },

                local => {
                    lock (locker) {
                        for (int i = 0; i < nbins; i++) {
                            histogram[i] += local[i];
                        }
                    }
                });

            int searchId = (int)Math.Round(histogram.Sum() * percentile);

            int idBin, acc = 0;
            for (idBin = 0; acc < searchId && idBin < nbins; idBin++) {

                acc += histogram[idBin];
            }

            return acc < searchId ? .03f : gradMax * idBin / nbins;
        }

        private static Image<Rgb24> ConvertToImage(float[] bufferData, int w, int h) {

            const int bpp = 3;
            float minVal = bufferData.Min();
            float maxVal = bufferData.Max();

            if (minVal == 0f && maxVal == 0f) {
                throw new ArgumentException("Layer data is empty!");
            }

            byte remap(float val) {

                float mapped = MathF.Round((val - minVal) / (maxVal - minVal) * 255f);
                return (byte)Math.Clamp(mapped, 0f, 255f);
            }

            Span<byte> binData = new byte[w * h * bpp];

            int k = 0;
            foreach (float inValue in bufferData) {

                byte outValue = remap(inValue);
                binData.Slice(k, bpp).Fill(outValue);
                k += bpp;
            }

            return Image.LoadPixelData<Rgb24>(binData, w, h);
        }

        private static void TestLoadImage(int i, Layer step, string directory) {

            void DownloadImage(float[]? buffer, string imgName) {

                ArgumentNullException.ThrowIfNull(buffer);
                using var img = ConvertToImage(buffer, step.Width, step.Height);
                img.SaveAsJpeg(directory + $@"\{imgName}.jpg");
            }

            DownloadImage(step.Lt, $"LT_{i}");
            DownloadImage(step.Lx, $"LX_{i}");
            DownloadImage(step.Ly, $"LY_{i}");
            DownloadImage(step.Ldet, $"LDET_{i}");
        }
    }
}
