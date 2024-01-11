using FeatureDetection;
using ILGPU;
using ILGPU.Runtime;
using System.Collections;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace FeatureDetectionConsoleApp {
    internal class Program {
        private static byte[] KeypointToByteArray(Keypoint kpt) {

            int kptSize = Marshal.SizeOf(kpt);
            var kptBin = new byte[kptSize];
            nint pointer = Marshal.AllocHGlobal(kptSize);
            try {
                Marshal.StructureToPtr(kpt, pointer, true);
                Marshal.Copy(pointer, kptBin, 0, kptSize);
            }
            finally {
                Marshal.FreeHGlobal(pointer);
            }
            return kptBin;
        }

        private static byte[] DescriptorToByteArray(BitArray descr) {

            int descrSize = (descr.Length - 1) / 8 + 1;
            var descrBin = new byte[descrSize];
            descr.CopyTo(descrBin, 0);
            return descrBin;
        }

        static void Main(string[] args) {

#if !DEBUG
            if (args.Length < 2
                || !Directory.Exists(args[0])
                || !Directory.Exists(args[1])) {

                Console.WriteLine("Error. This path does not exist");
                return;
            }

            string inputFolder = args[0];
            string outputFolder = args[1];
#else
            string inputFolder = @"C:\Users\danii\Documents\Photogrammetry\00 Input Dataset";
            string outputFolder = @"C:\Users\danii\Documents\Photogrammetry\01 Feature Detection";
#endif

            Console.WriteLine("Initializing...");

            Context context = Context.Create(i => i.Default().EnableAlgorithms());
            Device device = context.GetPreferredDevice(false);
            Accelerator accelerator = device.CreateAccelerator(context);
            var settings = new AKAZESettings();
            var akaze = new AKAZE(accelerator, settings);

            string[] inputFiles = Directory.GetFiles(inputFolder);
            var stopwatch = new Stopwatch();

            foreach (var inputFile in inputFiles) {

                using var img = new FileStream(inputFile, FileMode.Open, FileAccess.Read);
                stopwatch.Restart();
                akaze.GetKeypoints(img, out List<Keypoint> kpts, out List<BitArray> descr);
                stopwatch.Stop();

                if (kpts.Count == 0) {
                    Console.WriteLine("Points not found");
                    continue;
                }

                var inputFileInfo = new FileInfo(inputFile);
                string outputFileName = inputFileInfo.Name.Replace(inputFileInfo.Extension, ".kptdata");
                using var outputFileStream = new FileStream(outputFolder + @"\" + outputFileName, FileMode.Create, FileAccess.Write);
                using var writer = new BinaryWriter(outputFileStream);

                writer.Write((ulong)kpts.Count);
                writer.Write((ulong)descr[0].Length);

                foreach ((Keypoint keypoint, BitArray descriptor) in Enumerable.Zip(kpts, descr)) {

                    byte[] kptBin = KeypointToByteArray(keypoint);
                    writer.Write(kptBin);

                    byte[] descrBin = DescriptorToByteArray(descriptor);
                    writer.Write(descrBin);
                }

                Console.WriteLine("{0} ms, {1} point(s) found", stopwatch.ElapsedMilliseconds, kpts.Count);
            }

            Console.WriteLine("Done...");
        }
    }
}
