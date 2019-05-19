using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class Neuron
    {
        public List<Dendrite> InputDendrites { get; set; } = new List<Dendrite>();
        public List<Dendrite> OutputDendrites { get; set; } = new List<Dendrite>();
        public ActivationType ActivationType { get; private set; }
        //public double Gamma { get; set; }
        //public double Weight { get; private set; }
        //public double Bias { get; private set; }


        public Neuron(ActivationType activationType)
        {
            ActivationType = activationType;

            // Produces a random number between -0.5 and 0.5
            //Bias = (Generators.RandomGenerator.NextDouble() > 0.5 ? 1 : -1) *
            //    Generators.RandomGenerator.NextDouble() * 0.5;

            //Weight = (Generators.RandomGenerator.NextDouble() > 0.5 ? 1 : -1) *
            //    Generators.RandomGenerator.NextDouble() * 0.5;
        }

        public void ConfigureOutputDendrites()
        {
            // NOTE: Change ActivationType to class that holds hint for output dendrite count
            if (ActivationType == ActivationType.SoftMax)
            {
                foreach (var input in InputDendrites)
                {
                    OutputDendrites.Add(new Dendrite());
                }
            }
            else
            {
                OutputDendrites.Add(new Dendrite());
            }
        }

        // Feedforward
        public void Fire()
        {
            if (ActivationType == ActivationType.SoftMax)
            {
                var dendriteValues = new double[InputDendrites.Count];
                for (int i=0; i<InputDendrites.Count; i++)
                {
                    dendriteValues[i] = (InputDendrites[i].Signal.Value * InputDendrites[i].Weight) + InputDendrites[i].Bias;
                }
                var output = Activate.SoftMax(dendriteValues);
                for (int i = 0; i < OutputDendrites.Count; i++)
                {
                    OutputDendrites[i].Signal.Value = output[i];
                }
            }
            else
            {
                var coalescedValue = GetCoalescedSignal();

                foreach (var output in OutputDendrites)
                {
                    output.Signal.Value = Activate.GetActivation(ActivationType, coalescedValue);
                }
            }
        }

        //public void UpdateWeightAndBias()
        //{
        //    Bias -= Gamma * NetworkModel.LearningRate;
        //    foreach (var d in InputDendrites)
        //    {
        //        Weight -= Gamma * d.Signal.Value * NetworkModel.LearningRate;
        //    }
        //}

        private double GetCoalescedSignal()
        {
            double value = 0;

            foreach (var d in InputDendrites)
            {
                value += d.Signal.Value * d.Weight;
            }

            return value;
        }
    }
}
