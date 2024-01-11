using CommunityToolkit.HighPerformance;
using System.Collections;

namespace FeatureDetection {
    internal static class MLDB {
        private static void ComputeMeanValues(float[,,] samples, float sin, float cos, float[,,] mean, int subdiv, int subdivSz) {

            var samplesLt = new ReadOnlySpan2D<float>(samples, 0);
            var samplesLx = new ReadOnlySpan2D<float>(samples, 1);
            var samplesLy = new ReadOnlySpan2D<float>(samples, 2);

            int maxW = samplesLt.Width;
            int maxH = samplesLt.Height;

            var meanLt = new Span2D<float>(mean, 0);
            var meanLx = new Span2D<float>(mean, 1);
            var meanLy = new Span2D<float>(mean, 2);

            for (int i = 0; i < subdiv; i++) {
                for (int j = 0; j < subdiv; j++) {

                    int minX = j * subdivSz;
                    int minY = i * subdivSz;
                    int maxX = Math.Min((j + 1) * subdivSz, maxW);
                    int maxY = Math.Min((i + 1) * subdivSz, maxH);
                    meanLt[i, j] = meanLx[i, j] = meanLy[i, j] = 0f;

                    float sum = 0f;
                    for (int ii = minY; ii < maxY; ii++) {
                        for (int jj = minX; jj < maxX; jj++) {

                            meanLt[i, j] += samplesLt[ii, jj];
                            float dx = samplesLx[ii, jj];
                            float dy = samplesLy[ii, jj];
                            meanLy[i, j] += dx * cos + dy * sin;
                            meanLx[i, j] += dy * cos - dx * sin;
                            sum += 1f;
                        }
                    }

                    meanLt[i, j] /= sum;
                    meanLx[i, j] /= sum;
                    meanLy[i, j] /= sum;
                }
            }
        }

        private static void ComputeBinValues(float[,,] mean, BitArray desc, int subdiv, ref int outIndex) {

            var meanLt = new ReadOnlySpan2D<float>(mean, 0);
            var meanLx = new ReadOnlySpan2D<float>(mean, 1);
            var meanLy = new ReadOnlySpan2D<float>(mean, 2);

            int n = subdiv * subdiv;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {

                    int srcI = i / subdiv;
                    int srcJ = i % subdiv;
                    int dstI = j / subdiv;
                    int dstJ = j % subdiv;

                    desc[outIndex++] = meanLt[srcI, srcJ] > meanLt[dstI, dstJ];
                    desc[outIndex++] = meanLx[srcI, srcJ] > meanLx[dstI, dstJ];
                    desc[outIndex++] = meanLy[srcI, srcJ] > meanLy[dstI, dstJ];
                }
            }
        }

        public static BitArray ComputeDescriptor(Layer step, Keypoint kpt) {

            const int patternSize = 10;
            float invOctaveScale = 1f / (1 << kpt.Octave);
            float sigmaScale = MathF.Round(kpt.Size * invOctaveScale);

            const int descrBitSize = 486;
            var bits = new BitArray(descrBitSize);
            int n = 2 * patternSize + 1;
            var samples = new float[3, n, n];

            ReadOnlySpan2D<float> Lt = step.LtAsMatrix();
            ReadOnlySpan2D<float> Lx = step.LxAsMatrix();
            ReadOnlySpan2D<float> Ly = step.LyAsMatrix();

            (float sin, float cos) = MathF.SinCos(kpt.Angle);
            float curX = kpt.X * invOctaveScale;
            float curY = kpt.Y * invOctaveScale;

            for (int i = -patternSize; i <= patternSize; i++) {
                for (int j = -patternSize; j <= patternSize; j++) {

                    int x = (int)MathF.Round(curX + sigmaScale * (i * cos - j * sin));
                    int y = (int)MathF.Round(curY + sigmaScale * (j * cos + i * sin));

                    if (x < 0 || x >= Lt.Width || y < 0 || y >= Lt.Height) {
                        continue;
                    }
                    samples[0, i + patternSize, j + patternSize] = Lt[y, x];
                    samples[1, i + patternSize, j + patternSize] = Lx[y, x];
                    samples[2, i + patternSize, j + patternSize] = Ly[y, x];
                }
            }

            int outIndex = 0;

            int sz = patternSize;
            var sum = new float[3, sz, sz];
            ComputeMeanValues(samples, sin, cos, sum, 2, sz);
            ComputeBinValues(sum, bits, 2, ref outIndex);

            sz = (int)Math.Ceiling(2f * patternSize / 3f);
            sum = new float[3, sz, sz];
            ComputeMeanValues(samples, sin, cos, sum, 3, sz);
            ComputeBinValues(sum, bits, 3, ref outIndex);

            sz = patternSize / 2;
            sum = new float[3, sz, sz];
            ComputeMeanValues(samples, sin, cos, sum, 4, sz);
            ComputeBinValues(sum, bits, 4, ref outIndex);           

            return bits;
        }
    }
}
