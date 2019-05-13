using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class Neuron
    {
        public List<Dendrite> Dendrites { get; set; } = new List<Dendrite>();
        public ActivationType ActivationType { get; private set; }
        public double Gamma { get; set; }
        public double Weight { get; private set; }
        public double WeightDelta { get; private set; }
        public double Bias { get; private set; }
        public double BiasDelta { get; private set; }
        public Pulse Output { get; set; } = new Pulse();


        private static readonly Random _generator = new Random();


        public Neuron(ActivationType activationType, double? bias = null, double? weight = null)
        {
            ActivationType = activationType;

            if (bias.HasValue)
            {
                Bias = bias.Value;
            }
            else
            {
                // Produces a random number between -0.5 and 0.5
                Bias = (_generator.NextDouble() > 0.5 ? 1 : -1) *
                    _generator.NextDouble() * 0.5;
            }

            if (weight.HasValue)
            {
                Weight = weight.Value;
            }
            else
            {
                // Produces a random number between -0.5 and 0.5
                Weight = (_generator.NextDouble() > 0.5 ? 1 : -1) *
                    _generator.NextDouble() * 0.5;
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
            var prevBiasDelta = BiasDelta;
            BiasDelta = NetworkModel.LearningRate * Gamma;
            Bias += BiasDelta + NetworkModel.Momentum * prevBiasDelta;

            var prevWeightDelta = WeightDelta;
            WeightDelta = NetworkModel.LearningRate * Gamma * Output.Value;
            Weight += WeightDelta + NetworkModel.Momentum * prevWeightDelta;
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
