using CommunityToolkit.HighPerformance;
using ILGPU;
using ILGPU.Algorithms;
using ByteMatrix = ILGPU.Runtime.ArrayView2D<byte, ILGPU.Stride2D.DenseX>;
using IntMatrix = ILGPU.Runtime.ArrayView2D<int, ILGPU.Stride2D.DenseX>;

namespace FeatureMatchingConsoleApp {
    internal static class Tech {

        public static string GetKptDataFile(string kptDataFolder, string imgFilePath) {

            var imgFileInfo = new FileInfo(imgFilePath);
            return kptDataFolder + @"\" + imgFileInfo.Name.Replace(imgFileInfo.Extension, ".kptdata");
        }

        private static int[,] ConvertToBuffer(byte[][] input) {

            int intSz = sizeof(int);
            int IntArraySize(int n) => n % intSz == 0 ? n : IntArraySize(n + 1);

            int byteBufferWidth = IntArraySize(input[0].Length);
            int intBufferWidth = byteBufferWidth / intSz;

            int[] bytesToIntArray(byte[] bytes) {
                var intArray = new int[intBufferWidth];
                Buffer.BlockCopy(bytes, 0, intArray, 0, bytes.Length);
                return intArray;
            }

            var output = new int[input.Length, intBufferWidth];
            var outputSpan = new Span2D<int>(output);

            for (int i = 0; i < input.Length; i++) {

                Span<int> intArraySpan = bytesToIntArray(input[i]);
                Span<int> row = outputSpan.GetRowSpan(i);
                intArraySpan.CopyTo(row);
            }

            return output;
        }

        private static IEnumerable<byte[]> GetDescriptorsFromFile(string filePath) {

            using var fileStream = new FileStream(filePath, FileMode.Open);
            using var reader = new BinaryReader(fileStream);
            int descrCount = (int)reader.ReadUInt64();
            int descrSize = (int)reader.ReadUInt64();

            for (int i = 0; i < descrCount; i++) {

                reader.ReadBytes(28);
                byte[] descriptor = reader.ReadBytes(descrSize);
                yield return descriptor;
            }
        }

        public static int[,] ReadBuffer(string filePath) {

            var descriptors = GetDescriptorsFromFile(filePath);
            return ConvertToBuffer(descriptors.ToArray());
        }

        public static void BFMatcher(Index1D index, int threshold, IntMatrix descrA, IntMatrix descrB, ByteMatrix result) {

            for (int x = 0; x < descrB.IntExtent.X; x++) {

                int sum = 0;
                for (int y = 0; y < descrA.IntExtent.Y; y++) {
                    sum += XMath.PopCount(descrA[index.X, y] ^ descrB[x, y]);
                }

                result[index.X, x] = sum <= threshold ? (byte)1 : (byte)0;
            }
        }

        public static HashSet<(int, int)> GetUniquePairs(int n) {

            var pairs = new HashSet<(int, int)>();

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {

                    if (i == j)
                        continue;

                    if (pairs.Contains((i, j)) || pairs.Contains((j, i)))
                        continue;

                    pairs.Add((i, j));
                }

            return pairs;
        }
    }
}
