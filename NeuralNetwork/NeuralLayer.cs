using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class NeuralLayer
    {
        public List<Neuron> Neurons { get; private set; } = new List<Neuron>();
        public ActivationType ActivationType { get; private set; }
        public bool IsOutputLayer { get; private set; }


        public NeuralLayer(
            int neuronCount,
            ActivationType activationType = ActivationType.Relu,
            double? bias = null,
            double? weight = null, 
            bool isOutputLayer = false)
        {
            ActivationType = activationType;
            IsOutputLayer = isOutputLayer;

            for (int i = 0; i < neuronCount; i++)
            {
                Neurons.Add(new Neuron(ActivationType, bias, weight));
            }
        }

        public NeuralLayer(int neuronCount, ActivationType activationType = ActivationType.Relu)
        {
            ActivationType = activationType;

            for (int i=0; i<neuronCount; i++)
            {
                Neurons.Add(new Neuron(activationType));
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
