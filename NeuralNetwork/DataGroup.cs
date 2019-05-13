using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class DataGroup
    {
        public List<double[]> Values { get; private set; }
        public List<double> Labels { get; private set; }

        public DataGroup(IList<double[]> values, IList<double> labels)
        {
            if (values.Count != labels.Count)
                throw new Exception("Labels do not match values");

            Values = new List<double[]>(values);
            Labels = new List<double>(labels);
        }
    }
}
