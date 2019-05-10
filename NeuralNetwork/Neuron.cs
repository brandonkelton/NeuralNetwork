using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class Neuron
    {
        public List<Dendrite> Dendrites { get; set; } = new List<Dendrite>();
        public Pulse Output { get; set; } = new Pulse();

        public void Fire()
        {
            Output.Value = Activation(Sum());
        }

        public void UpdateWeights(double weight)
        {
            foreach (var d in Dendrites)
            {
                d.Weight = weight;
            }
        }

        private double Sum()
        {
            double value = 0.0f;

            foreach (var d in Dendrites)
            {
                value += d.Input.Value * d.Weight;
            }

            return value;
        }

        private double Activation(double value)
        {
            return Math.Tanh(value);
        }
    }
}
