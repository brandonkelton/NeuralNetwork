using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class NetworkModel
    {
        public static double LearningRate { get; set; } = 0.01;
        public static double Momentum { get; set; } = 1;

        public int Epochs { get; private set; }
        public List<NeuralLayer> Layers { get; private set; } = new List<NeuralLayer>();
        public double Cost { get; private set; }
        public double Accuracy { get; private set; }

        public NetworkModel(int epochs = 10)
        {
            Epochs = epochs;
        }

        public void AddLayer(
            int neuronCount,
            ActivationType activationType = ActivationType.Tanh)
        {
            var layer = new NeuralLayer(neuronCount, activationType);
            AddLayer(layer);
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
                    foreach (var lastLayerOutput in lastLayerNeuron.OutputDendrites)
                    {
                        var newLayerNeuronDendrite = new Dendrite { Signal = lastLayerOutput.Signal };
                        newLayerNeuron.InputDendrites.Add(newLayerNeuronDendrite);
                    }
                    // Each neuron in the new layer is assigned a dendrite
                    // corresponding to each neuron's output in the layer before.
                    //var newLayerNeuronDendrite = new Dendrite { Signal = lastLayerNeuron.Output };
                    //newLayerNeuron.InputDendrites.Add(newLayerNeuronDendrite);
                }
                newLayerNeuron.ConfigureOutputDendrites();
            }

            Layers.Add(layer);
        }

        //public void SetExpectedOutputs(
        //    double[] expectedOutputs, 
        //    ActivationType activationType = ActivationType.Tanh,
        //    double? bias = null,
        //    double? weight = null)
        //{
        //    _expectedOutputs = new double[expectedOutputs.Length];
        //    Array.Copy(expectedOutputs, _expectedOutputs, expectedOutputs.Length);
        //    var finalLayer = new NeuralLayer(expectedOutputs.Length, activationType, bias, weight, true);
        //    AddLayer(finalLayer);
        //}

        //public double[] GetExpectedOutputs()
        //{
        //    var expectedOutputs = new double[_expectedOutputs.Length];
        //    Array.Copy(_expectedOutputs, expectedOutputs, _expectedOutputs.Length);
        //    return expectedOutputs;
        //}

        private void FeedForward(double[] inputs)
        {
            if (Layers.Count < 1) throw new Exception("There are no layers!");

            UpdateFirstLayerInputs(inputs);

            foreach (var layer in Layers)
            {
                layer.Forward();
            }
        }

        private void UpdateFirstLayerInputs(double[] inputs)
        {
            // Add dendrites to the first layer according to the number of inputs,
            // but only if dendrites do not yet exist. An interesting twist on this
            // might be to allow the inputs to vary so, say, different image sizes
            // could be passed in. Not sure how well that would work, and it would
            // require some changes to the existing code (weights, biases handled at
            // network model level rather than Neuron instantiation).
            var firstLayer = Layers.First();
            if (firstLayer.Neurons.Sum(n => n.InputDendrites.Count()) < 1)
            {
                var dendritesPerNeuron = inputs.Count() / firstLayer.Neurons.Count();
                foreach (var neuron in Layers.First().Neurons)
                {
                    for (int i = 0; i < dendritesPerNeuron; i++)
                    {
                        var dendrite = new Dendrite();
                        neuron.InputDendrites.Add(dendrite);
                    }
                }
            }

            // Assign values to all first layer inputs
            var dendrites = firstLayer.Neurons.SelectMany(n => n.InputDendrites).ToArray();
            for (int i = 0; i < inputs.Count(); i++)
            {
                dendrites[i].Signal.Value = inputs[i];
            }
        }

        // Get the last layer's outputs and teach each layer
        private void BackPropagate(double[] labelList, double label)
        {
            var lastLayer = Layers.Last();
            // var lastLayerOutput = lastLayer.Neurons.Select(n => n.Output.Value).ToArray();

            //Cost = 0;
            //for (int i=0; i< expectedOutput.Count(); i++)
            //{
            //    Cost += Math.Pow(lastLayerOutput[i] - expectedOutput[i], 2);
            //}
            //Cost /= 2;

            // Build expected output - all neurons in last layer should be 
            // zero except the neuron corresponding to the label index
            var expectedOutput = Enumerable.Repeat(0.0d, labelList.Length).ToArray();
            expectedOutput[Array.IndexOf(labelList, label)] = 1.0d;

            lastLayer.Learn(expectedOutput);            

            // Teach all other layers, each based on the previous layer
            for (int i=Layers.Count - 2; i>0; i--)
            {
                Layers[i].Learn(Layers[i + 1]);
            }
        }

        public void Train(DataGroup trainGroup)
        {
            var results = new List<DetectionResult>();

            // Iterate through epochs and train network, keeping track of results
            for (int e=0; e<Epochs; e++)
            {
                for (int i=0; i<trainGroup.Values.Count; i++)
                {
                    FeedForward(trainGroup.Values[i]);
                    BackPropagate(trainGroup.LabelList, trainGroup.ValueLabels[i]);

                    // var label = trainGroup.ValueLabels[i];
                    var outputLayer = Layers.Last();
                    var outputResults = outputLayer.Neurons.SelectMany(n => n.OutputDendrites.Select(d => d.Signal.Value)).ToArray();
                    // var isCorrect = label.Equals(outputResult);

                    results.Add(new DetectionResult { Label = trainGroup.ValueLabels[i], OutputValues = outputResults });

                    if (i % 100 == 0)
                    {
                        var displayList = new List<string>();
                        for (int l=0; l< trainGroup.LabelList.Length; l++)
                        {
                            var label = trainGroup.LabelList[l];
                            var resultsList = results.Where(r => r.Label == label).ToList();
                            if (resultsList.Count > 0)
                            {
                                int correctCount = 0;
                                foreach (var result in resultsList)
                                {
                                    var maxValue = result.OutputValues.Max();
                                    var maxValueIndex = Array.IndexOf(result.OutputValues, maxValue);
                                    if (l == maxValueIndex)
                                    {
                                        correctCount++;
                                    }
                                }
                                var line = $"Label {label} Accuracy: {(correctCount / resultsList.Count) * 100}%";
                                displayList.Add(line);
                            }
                            else
                            {
                                displayList.Add($"Label {label} Accuracy: NOT YET CALCULATED");
                            }
                            
                        }

                        Console.WriteLine();
                        Console.WriteLine(String.Join("\n", displayList));
                    }
                    


                    //if (i % 100 == 0)
                    //{
                    //    Console.WriteLine();
                    //    foreach (var result in accuracyResults.OrderBy(r => r.RelevantKey).ToList())
                    //    {
                    //        Console.WriteLine($"{result.RelevantKey}\tAttempts: {result.Count}\tCorrect: {result.Correct}\tIncorrect: {result.Incorrect}\tAccuracy: {(result.Count == 0 ? 0 : (result.Correct / result.Count))}");
                    //    }
                    //}
                }
            }
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
