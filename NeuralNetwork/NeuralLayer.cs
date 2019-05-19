using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class NeuralLayer
    {
        public List<Neuron> Neurons { get; private set; } = new List<Neuron>();
        public ActivationType ActivationType { get; private set; }

        public NeuralLayer(int neuronCount, ActivationType activationType = ActivationType.Tanh)
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
                if (ActivationType == ActivationType.SoftMax)
                {
                    var valueDerivatives = Activate.SoftMaxDer(Neurons[i].OutputDendrites.Select(d => d.Signal.Value).ToArray());
                    for (int d = 0; d < Neurons[i].OutputDendrites.Count; d++)
                    {
                        Neurons[i].OutputDendrites[d].Gamma = (Neurons[i].OutputDendrites[d].Signal.Value - expectedOutput[d]) * valueDerivatives[d];
                        Neurons[i].OutputDendrites[d].UpdateWeightAndBias();
                    }
                }
                else
                {
                    //foreach (var dendrite in Neurons[i].OutputDendrites)
                    //{
                    //    Neurons[i].Gamma += (dendrite.Signal.Value - expectedOutput[i]) * Activate.GetActivationDer(ActivationType, dendrite.Signal.Value);
                    //}

                    for (int d = 0; d < Neurons[i].OutputDendrites.Count; d++)
                    {
                        var signal = Neurons[i].OutputDendrites[d].Signal;
                        var value = (signal.Value - expectedOutput[d]) * Activate.GetActivationDer(ActivationType, signal.Value);
                        Neurons[i].OutputDendrites[d].Gamma = value;
                        Neurons[i].OutputDendrites[d].UpdateWeightAndBias();
                    }
                }
                

                //var averageDendriteOutput = Neurons[i].OutputDendrites.Average(d => d.Signal.Value);

                //Neurons[i].Gamma = 
                //    (averageDendriteOutput - expectedOutput[i]) *
                //        Activate.GetActivationDer(ActivationType, averageDendriteOutput);

                //Neurons[i].UpdateWeightAndBias();
            }
        }

        public void Learn(NeuralLayer fromLayer)
        {
            for (int i=0; i < Neurons.Count; i++)
            {
                var valueDerivatives = Activate.SoftMaxDer(Neurons[i].OutputDendrites.Select(d => d.Signal.Value).ToArray());
                for (int d = 0; d < Neurons[i].OutputDendrites.Count; d++)
                {
                    var currentDendrite = Neurons[i].OutputDendrites[d];
                    foreach (var fromNeuron in fromLayer.Neurons)
                    {
                        foreach (var fromDendrite in fromNeuron.InputDendrites)
                        {
                            currentDendrite.Gamma += fromDendrite.Gamma * currentDendrite.Weight;
                        }
                    }

                    currentDendrite.Gamma *= ActivationType == ActivationType.SoftMax 
                        ? valueDerivatives[d] 
                        : Activate.GetActivationDer(ActivationType, currentDendrite.Signal.Value);

                    currentDendrite.UpdateWeightAndBias();
                }

                //foreach (var fromNeuron in fromLayer.Neurons)
                //{
                //    Neurons[i].Gamma += fromNeuron.Gamma * Neurons[i].Weight;
                //}

                //var averageDendriteOutput = Neurons[i].OutputDendrites.Average(d => d.Signal.Value);
                //Neurons[i].Gamma *= Activate.GetActivationDer(ActivationType, averageDendriteOutput);
                //Neurons[i].UpdateWeightAndBias();
            }
        }

        public void Log()
        {
            //Console.WriteLine($"Layer: {LayerLevel}\tNeurons: {Neurons.Count}\tWeight: {Weight}");
        }
    }
}
