using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class Dendrite
    {
        public Pulse Signal { get; set; } = new Pulse();

        public double Gamma { get; set; }

        public double Weight { get; set; }

        public double Bias { get; set; }

        public Dendrite()
        {
            Weight = (Generators.RandomGenerator.NextDouble() > 0.5 ? 1 : -1) *
                Generators.RandomGenerator.NextDouble() * 0.5;

            Bias = (Generators.RandomGenerator.NextDouble() > 0.5 ? 1 : -1) *
                Generators.RandomGenerator.NextDouble() * 0.5;
        }

        public void UpdateWeightAndBias()
        {
            Bias -= Gamma * NetworkModel.LearningRate;
            Weight -= Gamma * Signal.Value * NetworkModel.LearningRate;
        }
    }
}
