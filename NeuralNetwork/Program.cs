using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            var dataGroup = LoadTrainingData();

            NetworkModel.LearningRate = 0.01d;
            var model = new NetworkModel(epochs: 20);
            model.AddLayer(dataGroup.Values[0].Length);
            model.AddLayer(512);
            model.AddLayer(256);
            model.AddLayer(dataGroup.LabelList.Length);
            model.AddLayer(1, ActivationType.SoftMax);

            model.Train(dataGroup);

            Console.WriteLine("\n\nTraining Complete!");
            Console.ReadLine();
        }

        static DataGroup LoadTrainingData()
        {
            var directoryPath = Directory.GetCurrentDirectory();

            var file = new FileInfo($"{directoryPath}\\Resources\\number-image-train.csv");
            if (!file.Exists) throw new Exception($"File Doesn't Exist:  {file.FullName}");

            var rows = new List<double[]>();
            var labels = new List<double>();

            using (var reader = new StreamReader(file.FullName))
            {
                var count = 0;
                while (!reader.EndOfStream)
                {
                    count++;

                    var line = reader.ReadLine();
                    if (count == 1) continue; // Skip first line that has column names
                    var data = line.Split(",");
                    var rowData = new double[data.Length];

                    string label = data[0];
                    double convertedLabel;
                    if (!double.TryParse(label, out convertedLabel))
                    {
                        convertedLabel = 0.0d;
                    }
                    labels.Add(convertedLabel);

                    for (int i=1; i<data.Length; i++)
                    {
                        string dataPoint = data[i];
                        double convertedDataPoint;
                        if (!double.TryParse(dataPoint, out convertedDataPoint))
                        {
                            convertedDataPoint = 0.0d;
                        }
                        rowData[i] = convertedDataPoint;
                    }

                    rows.Add(rowData);
                    count++;
                }
            }

            var labelList = labels.Distinct().OrderBy(l => l).ToArray();

            //var rowLabels = new List<double[]>(labelList.Length);
            //foreach (var label in labels)
            //{
            //    var expectedOutput = Enumerable.Repeat(-1.0d, labelList.Length).ToArray();
            //    for (int i = 0; i < labelList.Length; i++)
            //    {
            //        if (labelList[i] == label)
            //        {
            //            expectedOutput[i] = label;
            //        }
            //    }
            //    rowLabels.Add(expectedOutput);
            //}

            //var possibleNeuronOutputs = new List<double[]>(labelList.Length);
            //for (int i = 0; i < labelList.Count(); i++)
            //{
            //    var result = Enumerable.Repeat(-1.0d, labelList.Count()).ToArray();
            //    for (int j = 0; j < labelList.Count(); j++)
            //    {
            //        if (j == i)
            //        {
            //            result[j] = labelList[j];
            //        }
            //    }
            //    possibleNeuronOutputs.Add(result);
            //}

            var dataGroup = new DataGroup(labelList, rows.ToArray(), labels.ToArray());
            return dataGroup;
        }
    }
}
