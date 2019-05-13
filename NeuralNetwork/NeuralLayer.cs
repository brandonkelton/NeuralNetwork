using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class NeuralLayer
    {
        public List<Neuron> Neurons { get; private set; } = new List<Neuron>();
        public ActivationType ActivationType { get; private set; }
        public double LearningRate { get; private set; }
        public bool IsOutputLayer { get; private set; }


        public NeuralLayer(
            int neuronCount,
            ActivationType activationType = ActivationType.Relu,
            double learningRate = 0.01,
            double? bias = null,
            double? weight = null, 
            bool isOutputLayer = false)
        {
            ActivationType = activationType;
            IsOutputLayer = isOutputLayer;
            LearningRate = learningRate;

            for (int i = 0; i < neuronCount; i++)
            {
                Neurons.Add(new Neuron(ActivationType, LearningRate, bias, weight));
            }
        }

        //public double Weight { get; private set; }

        //public NeuralLayer(int layerLevel, double initialWeight, int count)
        //{
        //    LayerLevel = layerLevel;
        //    Weight = initialWeight;

        //    for (int i=0; i<count; i++)
        //    {
        //        Neurons.Add(new Neuron());
        //    }
        //}

        //public void Optimize(double learningRate, double delta)
        //{
        //    Weight += learningRate * delta;
        //    foreach (var neuron in Neurons)
        //    {
        //        neuron.UpdateWeights(Weight);
        //    }
        //}

        public NeuralLayer(int neuronCount, ActivationType activationType, double learningRate)
        {
            ActivationType = activationType;
            LearningRate = learningRate;

            for (int i=0; i<neuronCount; i++)
            {
                Neurons.Add(new Neuron(activationType, learningRate));
            }
        }

        public void Forward()
        {
            foreach (var neuron in Neurons)
            {
                neuron.Fire();
            }
        }

        public void Learn(double[] expectedOutput)
        {
            for (int i = 0; i < Neurons.Count; i++)
            {
                // Set Gamma according to expected output values
                Neurons[i].Gamma =
                    (Neurons[i].Output.Value - expectedOutput[i]) *
                        Activate.GetActivationDer(ActivationType, Neurons[i].Output.Value);

                Neurons[i].UpdateWeightsAndBiases();
            }
        }

        public void Learn(NeuralLayer propagateFromLayer)
        {
            foreach (var neuron in Neurons)
            {
                foreach (var propFromNeuron in propagateFromLayer.Neurons)
                {
                    neuron.Gamma += propFromNeuron.Gamma * neuron.Weight;
                }

                neuron.Gamma *= Activate.GetActivationDer(ActivationType, neuron.Output.Value);
                neuron.UpdateWeightsAndBiases();
            }
        }

        public void Log()
        {
            //Console.WriteLine($"Layer: {LayerLevel}\tNeurons: {Neurons.Count}\tWeight: {Weight}");
        }
    }
}
