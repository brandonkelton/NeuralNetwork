using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class NeuralLayer : INeuralLayer
    {
        public int LayerLevel { get; private set; }
        public List<Neuron> Neurons { get; private set; } = new List<Neuron>();
        public double Weight { get; private set; }

        public NeuralLayer(int layerLevel, double initialWeight, int count)
        {
            LayerLevel = layerLevel;
            Weight = initialWeight;

            for (int i=0; i<count; i++)
            {
                Neurons.Add(new Neuron());
            }
        }

        public void Optimize(double learningRate, double delta)
        {
            Weight += learningRate * delta;
            foreach (var neuron in Neurons)
            {
                neuron.UpdateWeights(Weight);
            }
        }

        public void Forward()
        {
            foreach (var neuron in Neurons)
            {
                neuron.Fire();
            }
        }

        public void Log()
        {
            Console.WriteLine($"Layer: {LayerLevel}\tNeurons: {Neurons.Count}\tWeight: {Weight}");
        }
    }
}
