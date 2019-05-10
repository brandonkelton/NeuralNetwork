using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class NetworkModel
    {
        public float LearningRate { get; private set; }
        public int Epochs { get; private set; }
        public List<INeuralLayer> Layers { get; private set; } = new List<INeuralLayer>();

        public NetworkModel(float learningRate = 0.01f, int epochs = 10)
        {
            LearningRate = learningRate;
            Epochs = epochs;
        }

        public void AddLayer(INeuralLayer layer)
        {
            int dendriteCount = 1;

            if (Layers.Count > 0)
            {
                dendriteCount = Layers.Last().Neurons.Count;
            }

            // May not need this, but it could be some sort of bias
            foreach (var neuron in layer.Neurons)
            {
                for (int i=0; i<dendriteCount; i++)
                {
                    neuron.Dendrites.Add(new Dendrite());
                }
            }
        }

        public void CreateNetwork(INeuralLayer layerFrom, INeuralLayer layerTo)
        {
            foreach (var neuronFrom in layerFrom.Neurons)
            {
                foreach (var neuronTo in layerTo.Neurons)
                {
                    var dendrite = new Dendrite
                    {
                        Input = neuronFrom.Output,
                        Weight = layerTo.Weight
                    };
                    neuronTo.Dendrites.Add(dendrite);
                }
            }
        }

        public void Build()
        {
            if (Layers.Count < 2) return;

            for (int i=0; i<(i-1); i++)
            {
                CreateNetwork(Layers[i], Layers[i + 1]);
            }
        }

        private void ComputeOutput()
        {
            if (Layers.Count < 2) return;

            var layers = Layers.Skip(1).ToList();
            foreach (var layer in layers)
            {
                layer.Forward();
            }
        }

        private void OptimizeWeights(double accuracy)
        {
            if (accuracy == 1) return;

            var rate = accuracy > 1 ? -LearningRate : LearningRate;
            foreach (var layer in Layers)
            {
                layer.Optimize(rate, 1);
            }

        }

        public void Train()
        {

        }

        public void Log()
        {
            foreach (var layer in Layers)
            {
                layer.Log();
            }
        }
    }
}
