using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public static class Activate
    {
        public static double Sigmoid(double value)
        {
            var ePow = Math.Exp(value);
            var result = ePow / (1.0f + ePow);
            return result;
        }

        public static double Tanh(double value)
        {
            var result = Math.Tanh(value);
            return result;
        }

        public static double Relu(double value)
        {
            var result = (0 >= value) ? 0 : value;
            return result;
        }

        public static double LeakyRelu(double value)
        {
            var result = (0 > -value) ? 0.01f * value : value;
            return result;
        }

        public static double SigmoidDer(double value)
        {
            var result = value * (1 - value);
            return result;
        }

        public static double TanhDer(double value)
        {
            var result = 1 - (value * value);
            return result;
        }

        public static double ReluDer(double value)
        {
            var result = (0 >= value) ? 0 : 1;
            return result;
        }

        public static double LeakyReluDer(double value)
        {
            var result = (0 >= value) ? 0.01f : 1;
            return result;
        }

        public static double GetActivation(ActivationType activationType, double value)
        {
            switch (activationType)
            {
                case ActivationType.LeakyRelu:
                    return LeakyRelu(value);
                case ActivationType.Relu:
                    return Relu(value);
                case ActivationType.Sigmoid:
                    return Sigmoid(value);
                case ActivationType.Tanh:
                    return Tanh(value);
                default:
                    throw new Exception("Invalid ActivationType");
            }
        }

        public static double GetActivationDer(ActivationType activationType, double value)
        {
            switch (activationType)
            {
                case ActivationType.LeakyRelu:
                    return LeakyReluDer(value);
                case ActivationType.Relu:
                    return ReluDer(value);
                case ActivationType.Sigmoid:
                    return SigmoidDer(value);
                case ActivationType.Tanh:
                    return TanhDer(value);
                default:
                    throw new Exception("Invalid ActivationType");
            }
        }
    }
}
