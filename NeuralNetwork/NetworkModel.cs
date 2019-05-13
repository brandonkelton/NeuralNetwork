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
        public List<NeuralLayer> Layers { get; private set; } = new List<NeuralLayer>();

        private List<double> _expectedOutputs = null;

        public double Cost { get; private set; }

        public NetworkModel(float learningRate = 0.01f, int epochs = 10)
        {
            LearningRate = learningRate;
            Epochs = epochs;
        }

        public void AddLayer(NeuralLayer layer)
        {
            // Don't give the first layer any dendrites yet. They will
            // be determined on FeedForward by the number of inputs.
            // Normally, you will want to have the same number of neurons as
            // inputs, but this adds flexibility for trying out different approaches.
            if (Layers.Count == 0)
            {
                Layers.Add(layer);
                return;
            }

            var currentLastLayer = Layers.Last();
            foreach (var newLayerNeuron in layer.Neurons)
            {
                foreach (var lastLayerNeuron in currentLastLayer.Neurons)
                {
                    // Each neuron in the new layer is assigned a dendrite
                    // corresponding to each neuron's output in the layer before.
                    var newLayerNeuronDendrite = new Dendrite();
                    newLayerNeuronDendrite.Input = lastLayerNeuron.Output;
                    newLayerNeuron.Dendrites.Add(newLayerNeuronDendrite);
                }
            }
        }

        public void SetExpectedOutputs(
            double[] expectedOutputs, 
            ActivationType activationType,
            double? bias = null,
            double? weight = null)
        {
            _expectedOutputs = new List<double>(expectedOutputs);
            var finalLayer = new NeuralLayer(expectedOutputs.Count(), activationType, LearningRate, bias, weight, true);
            AddLayer(finalLayer);
        }

        public List<double> GetExpectedOutputs()
        {
            var expectedOutputs = new double[_expectedOutputs.Count];
            _expectedOutputs.CopyTo(expectedOutputs);
            return expectedOutputs.ToList();
        }

        //public void CreateNetwork(NeuralLayer layerFrom, NeuralLayer layerTo)
        //{
        //    foreach (var neuronFrom in layerFrom.Neurons)
        //    {
        //        foreach (var neuronTo in layerTo.Neurons)
        //        {
        //            var dendrite = new Dendrite
        //            {
        //                Input = neuronFrom.Output,
        //                Weight = layerTo.Weight
        //            };
        //            neuronTo.Dendrites.Add(dendrite);
        //        }
        //    }
        //}

        //public void Build()
        //{
        //    if (Layers.Count < 2) return;

        //    for (int i=0; i<(i-1); i++)
        //    {
        //        CreateNetwork(Layers[i], Layers[i + 1]);
        //    }
        //}

        //private void ComputeOutput()
        //{
        //    if (Layers.Count < 2) return;

        //    var layers = Layers.Skip(1).ToList();
        //    foreach (var layer in layers)
        //    {
        //        layer.Forward();
        //    }
        //}

        // should use back propagation instead
        //private void OptimizeWeights(double accuracy)
        //{
        //    if (accuracy == 1) return;

        //    // var rate = accuracy > 1 ? -LearningRate : LearningRate;
        //    foreach (var layer in Layers)
        //    {
        //        layer.Optimize(LearningRate, 1);
        //    }

        //}

        public void FeedForward(double[] inputs)
        {
            if (Layers.Count < 1) throw new Exception("There are no layers!");

            // Add dendrites to the first layer according to the number of inputs,
            // but only if dendrites do not yet exist. An interesting twist on this
            // might be to allow the inputs to vary so, say, different images sizes
            // could be passed in. Not sure how well that would work, and it would
            // require some changes to the existing code (weights, biases handled at
            // network model level rather than Dendrite instantiation).
            var firstLayer = Layers.First();
            if (firstLayer.Neurons.Sum(n => n.Dendrites.Count()) < 1)
            {
                var dendritesPerNeuron = inputs.Count() / firstLayer.Neurons.Count();
                foreach (var neuron in Layers.First().Neurons)
                {
                    for (int i=0; i<dendritesPerNeuron; i++)
                    {
                        var dendrite = new Dendrite();
                        neuron.Dendrites.Add(dendrite);
                    }
                }
            }

            var dendrites = firstLayer.Neurons.SelectMany(n => n.Dendrites).ToArray();
            for (int i=0; i<inputs.Count(); i++)
            {
                dendrites[i].Input.Value = inputs[i];
            }

            foreach (var layer in Layers)
            {
                layer.Forward();
            }
        }

        // Get the last layer's outputs and teach each layer
        public void BackPropagate(double[] expectedOutput)
        {
            var lastLayer = Layers.Last();
            var lastLayerOutput = lastLayer.Neurons.Select(n => n.Output.Value).ToArray();

            Cost = 0;
            for (int i=0; i< expectedOutput.Count(); i++)
            {
                Cost += Math.Pow(lastLayerOutput[i] - expectedOutput[i], 2);
            }
            Cost /= 2;

            // Teach last layer
            lastLayer.Learn(expectedOutput);            

            // Teach all other layers, which is based on the last layer's learning
            for (int i=Layers.Count - 2; i>0; i--)
            {
                Layers[i].Learn(Layers[i + 1]);
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
