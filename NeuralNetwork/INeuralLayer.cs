using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public interface INeuralLayer
    {
        int LayerLevel { get; }
        List<Neuron> Neurons { get; }
        double Weight { get; }

        void Optimize(double learningRate, double delta);
        void Forward();
        void Log();
    }
}
