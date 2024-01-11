using ILGPU;
using ILGPU.Algorithms;
using ByteArray = ILGPU.Runtime.ArrayView1D<byte, ILGPU.Stride1D.Dense>;
using FArray = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;
using FMatrix = ILGPU.Runtime.ArrayView2D<float, ILGPU.Stride2D.DenseX>;

namespace FeatureDetection {
    internal static class GPU {
        public static void CentralFED(Index2D index, FMatrix src, FMatrix diff, float t, FMatrix output) {

            if (index.X <= 0 || index.X >= src.IntExtent.X - 1 || index.Y <= 0 || index.Y >= src.IntExtent.Y - 1) {
                output[index] = 0f;
            }
            else {
                float curSrc = src[index];
                float curDiff = diff[index];
                output[index] =
                    t * (((curDiff + diff[index.X + 1, index.Y]) * (src[index.X + 1, index.Y] - curSrc))
                    - ((curDiff + diff[index.X - 1, index.Y]) * (curSrc - src[index.X - 1, index.Y]))
                    + ((curDiff + diff[index.X, index.Y + 1]) * (src[index.X, index.Y + 1] - curSrc))
                    - ((curDiff + diff[index.X, index.Y - 1]) * (curSrc - src[index.X, index.Y - 1])));
            }
        }

        public static void ComputeGradient(Index2D index, FMatrix Lx, FMatrix Ly, FMatrix grad) {

            float lx = Lx[index];
            float ly = Ly[index];
            grad[index] = XMath.Sqrt(lx * lx + ly * ly);
        }

        public static void ComputeHessian(Index2D index, FMatrix Lxx, FMatrix Lxy, FMatrix Lyy, float sigma, FMatrix output) {

            float lxx = Lxx[index];
            float lxy = Lxy[index];
            float lyy = Lyy[index];
            output[index] = (lxx * lyy - lxy * lxy) * sigma;
        }

        public static void Halfsize(Index2D index, FMatrix input, FMatrix output) {

            double x0 = 2d * (index.X + .5);
            double y0 = 2d * (index.Y + .5);
            double dx = x0 - XMath.Floor(x0);
            double dy = y0 - XMath.Floor(y0);
            const int samplerSize = 2;
            int halfSize = samplerSize / 2;

            double[] coefsX = new double[samplerSize];
            double[] coefsY = new double[samplerSize];

            coefsX[0] = 1d - dx;
            coefsX[1] = dx;
            coefsY[0] = 1d - dy;
            coefsY[1] = dy;

            double res = 0d;
            double totalWeight = 0d;
            int gridX = (int)XMath.Floor(x0);
            int gridY = (int)XMath.Floor(y0);

            for (int i = 0; i < samplerSize; i++) {

                int curI = gridX + 1 + i - halfSize;
                if (curI < 0 || curI >= input.IntExtent.X)
                    continue;

                for (int j = 0; j < samplerSize; j++) {

                    int curJ = gridY + 1 + j - halfSize;
                    if (curJ < 0 || curJ >= input.IntExtent.Y)
                        continue;

                    double w = coefsX[j] * coefsY[i];
                    double pix = (double)input[curI, curJ];
                    res += pix * w;
                    totalWeight += w;
                }
            }

            double result = totalWeight <= .2
                ? 0d
                : totalWeight != 1d
                    ? res / totalWeight
                    : res;

            output[index] = (float)result;
        }

        public static void HorizontalConvolution(Index2D index, FMatrix input, FArray kernel, FMatrix output) {

            float sum = 0f;
            int pad = kernel.IntLength / 2;

            for (int j = 0; j < kernel.IntLength; j++) {

                float src = index.X + j < pad
                    ? input[0, index.Y]
                    : index.X + j >= input.IntExtent.X + pad
                        ? input[input.IntExtent.X - 1, index.Y]
                        : input[index.X + j - pad, index.Y];

                sum += src * kernel[j];
            }

            if (index.X < input.IntExtent.X)
                output[index] = sum;
        }

        public static void PeronaMalikG2Diff(Index2D index, FMatrix Lx, FMatrix Ly, float k2, FMatrix diff) {

            float lx = Lx[index];
            float ly = Ly[index];
            diff[index] = 1f / (1f + (lx * lx + ly * ly) / k2);
        }

        public static void ToGrayscale(Index2D index, ByteArray src, int bpp, int stride, FMatrix output) {

            const float maxVal = 255f;
            int i = stride * index.Y + bpp * index.X;

            float R = src[i] / maxVal;
            float G = src[i + 1] / maxVal;
            float B = src[i + 2] / maxVal;

            float min = XMath.Min(R, XMath.Min(G, B));
            float max = XMath.Max(R, XMath.Max(G, B));
            output[index] = (max + min) / 2f;
        }

        public static void UpdateSrc(Index2D index, FMatrix src, FMatrix upd) => src[index] += upd[index];

        public static void VerticalConvolution(Index2D index, FMatrix input, FArray kernel, FMatrix output) {

            float sum = 0f;
            int pad = kernel.IntLength / 2;

            for (int j = 0; j < kernel.IntLength; j++) {

                float src = index.Y + j < pad
                    ? input[index.X, 0]
                    : index.Y + j >= input.IntExtent.Y + pad
                        ? input[index.X, input.IntExtent.Y - 1]
                        : input[index.X, index.Y + j - pad];

                sum += src * kernel[j];
            }

            if (index.Y < input.IntExtent.Y)
                output[index] = sum;
        }
    }
}