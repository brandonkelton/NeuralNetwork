using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class DetectionResult
    {
        public double Label { get; set; }

        public double[] OutputValues { get; set; }
    }
}
