using CommunityToolkit.HighPerformance;
using ILGPU.Runtime;
using ILGPU;
using System.Diagnostics;
using ByteMatrix = ILGPU.Runtime.ArrayView2D<byte, ILGPU.Stride2D.DenseX>;
using IntMatrix = ILGPU.Runtime.ArrayView2D<int, ILGPU.Stride2D.DenseX>;

namespace FeatureMatchingConsoleApp {
    internal class Program {
        private const string descrFolder = @"C:\Users\danii\Documents\Photogrammetry\01 Feature Detection";
        private const string imgFolder = @"C:\Users\danii\Documents\Photogrammetry\00 Input Dataset";
        private const string outputFolder = @"C:\Users\danii\Documents\Photogrammetry\02 Feature Matching";
        static void Main(string[] args) {
            Console.WriteLine("Initializing...");

            Context context = Context.Create(i => i.Default().EnableAlgorithms());
            Device device = context.GetPreferredDevice(false);
            Accelerator accelerator = device.CreateAccelerator(context);

            int[][,] buffers =
                 Directory.GetFiles(imgFolder)
                .Select(file => {
                    string bufferFile = Tech.GetKptDataFile(descrFolder, file);
                    return Tech.ReadBuffer(bufferFile);
                }).ToArray();

            var bfmatch = accelerator.LoadAutoGroupedStreamKernel<Index1D, int, IntMatrix, IntMatrix, ByteMatrix>(Tech.BFMatcher);

            IEnumerable<(int, int)> pairs = Tech.GetUniquePairs(buffers.Length);
            var locker = new object();
            using var outputFile = new FileStream(outputFolder + @"\output.txt", FileMode.Create);
            using var writer = new StreamWriter(outputFile);
            var stopwatch = new Stopwatch();
            stopwatch.Start();
            var start = DateTime.Now;

            foreach ((int a, int b) in pairs) {

                using var bufA = accelerator.Allocate2DDenseX(buffers[a]);
                using var bufB = accelerator.Allocate2DDenseX(buffers[b]);
                var resIndex = new Index2D(bufA.IntExtent.X, bufB.IntExtent.X);
                using var resBuf = accelerator.Allocate2DDenseX<byte>(resIndex);
                var index = new Index1D(bufA.IntExtent.X);

                bfmatch(index, 40, bufA.View, bufB.View, resBuf.View);
                accelerator.DefaultStream.Synchronize();

                const int maxChunkSize = 1000;
                int N = (int)Math.Ceiling((double)resIndex.Y / maxChunkSize);
                var pairIndices = new List<(int, int)>();

                for (int k = 0; k < N; k++) {

                    int chunkSize = k == N - 1 ? resIndex.Y - k * maxChunkSize : maxChunkSize;
                    var offset = new Index2D(0, k * maxChunkSize);
                    var extent = new Index2D(resIndex.X, chunkSize);
                    byte[] bufChunk = resBuf.View.SubView(offset, extent).To1DView().GetAsArray1D();
                    var buffer = new ReadOnlyMemory2D<byte>(bufChunk, chunkSize, resIndex.X);

                    Parallel.For(
                        0,
                        buffer.Height,

                        () => new List<(int, int)>(),

                        (i, _, local) => {
                            ReadOnlySpan2D<byte> resSpan = buffer.Span;

                            for (int j = 0; j < resSpan.Width; j++)
                                if (resSpan[i, j] == 1)
                                    pairIndices.Add((j, i + k * maxChunkSize));

                            return local;
                        },

                        local => {
                            if (local.Count != 0)
                                lock (locker) pairIndices.AddRange(local);
                        });
                }

                if (pairIndices.Count < 8) {

                    stopwatch.Restart();
                    continue;
                }

                writer.Write(a);
                writer.Write(" ");
                writer.Write(b);
                writer.Write(" ");

                string indices = pairIndices.Aggregate("", (txt, pair) => txt + $"{pair.Item1} {pair.Item2} ");
                writer.WriteLine(indices);

                stopwatch.Stop();
                Console.WriteLine("{0} ms, (#{1} - #{2}), {3} matches found", stopwatch.ElapsedMilliseconds, a, b, pairIndices.Count);
                stopwatch.Restart();
            }

            var end = DateTime.Now;
            Console.WriteLine("{0}, Done...", end - start);
        }
    }
}
