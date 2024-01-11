using FeatureDetection;
using ILGPU;
using ILGPU.Runtime;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace PhotogrammetryTests {
    [TestClass]
    public class FeatureDetectionTests {

        [TestMethod]
        public void PyramidTest() {

            const string inputFolder = @"C:\Users\danii\Documents\Photogrammetry\00 Input Dataset\IMG_0055.JPG";
            const string outputFolder = @"C:\Users\danii\Documents\Photogrammetry\00_1 Test Data";

            Context context = Context.Create(i => i.Default().EnableAlgorithms());
            Device device = context.GetPreferredDevice(false);
            Accelerator accelerator = device.CreateAccelerator(context);

            using var img = new FileStream(inputFolder, FileMode.Open);
            var akaze = new AKAZE(accelerator, new AKAZESettings());

            try {
                akaze.TestPyramid(img, outputFolder);
            }
            catch (Exception ex) {
                Assert.Fail(ex.Message);
            }
        }

        [TestMethod]
        public void DetectionTest() {

            const string inputFolder = @"C:\Users\danii\Documents\Photogrammetry\00 Input Dataset\IMG_0055.JPG";
            Context context = Context.Create(i => i.Default().EnableAlgorithms());
            Device device = context.GetPreferredDevice(false);
            Accelerator accelerator = device.CreateAccelerator(context);

            using var img = new FileStream(inputFolder, FileMode.Open);
            var akaze = new AKAZE(accelerator, new AKAZESettings());            
            akaze.GetKeypoints(img, out List<Keypoint> kpts, out _);
            Assert.IsTrue(kpts.Any());
        }

        [TestMethod]
        public void SolveEquationTest() {

            Matrix<float> A = Matrix<float>.Build.DenseOfRowMajor(2, 2, [1f, -2.5f, .3f, 1]);
            Vector<float> B = Vector<float>.Build.DenseOfArray([7.5f, 1f]);
            var X = A.Solve(B).Select(i => i.Round(3));

            Assert.IsTrue(X.SequenceEqual([5.714f, -0.714f]));
        }
    }
}