using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class DataGroup
    {
        public double[] LabelList { get; private set; }
        public List<double[]> Values { get; private set; }
        public double[] ValueLabels { get; private set; }

        public DataGroup(double[] labelList, IList<double[]> values, double[] valueLabels)
        {
            if (values.Count != valueLabels.Length)
                throw new Exception("Labels do not match values");

            LabelList = labelList;
            Values = new List<double[]>(values);
            ValueLabels = valueLabels;
        }
    }
}
