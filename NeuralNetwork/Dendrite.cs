using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class Dendrite
    {
        public Pulse Input { get; set; }

        public double Weight { get; set; }

        public bool CanLearn { get; set; } = true;
    }
}
