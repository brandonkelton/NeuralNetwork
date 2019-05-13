using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class Neuron
    {
        private static Random Generator = new Random();

        public List<Dendrite> Dendrites { get; set; } = new List<Dendrite>();
        public ActivationType ActivationType { get; private set; }
        public double LearningRate { get; private set; }
        public double Gamma { get; set; }
        public bool CanLearn { get; set; }
        public double Weight { get; set; }
        public double Bias { get; set; }
        public Pulse Output { get; set; } = new Pulse();


        public Neuron(ActivationType activationType, double learningRate, double? bias = null, double? weight = null)
        {
            ActivationType = activationType;
            LearningRate = learningRate;

            if (bias.HasValue)
            {
                Bias = bias.Value;
            }
            else
            {
                // Produces a random number between -0.5 and 0.5
                Bias = (Generator.NextDouble() > 0.5 ? 1 : -1) *
                Generator.NextDouble() * 0.5;
            }

            if (weight.HasValue)
            {
                Weight = weight.Value;
            }
            else
            {
                // Produces a random number between -0.5 and 0.5
                Weight = (Generator.NextDouble() > 0.5 ? 1 : -1) *
                    Generator.NextDouble() * 0.5;
            }
        }

        // Feedforward
        public void Fire()
        {
            var coalescedValue = GetCoalescedSignal();
            Output.Value = ActivateValue(coalescedValue);
        }

        public void UpdateWeightsAndBiases()
        {
            if (!CanLearn) return;
            Bias -= Gamma * LearningRate;
            Weight -= Gamma * Output.Value * LearningRate;
        }

        private double GetCoalescedSignal()
        {
            double value = 0;

            foreach (var d in Dendrites)
            {
                value += (d.Input.Value * Weight) + Bias;
            }

            return value;
        }

        private double ActivateValue(double value)
        {
            return Activate.GetActivation(ActivationType, value);
        }
    }
}
