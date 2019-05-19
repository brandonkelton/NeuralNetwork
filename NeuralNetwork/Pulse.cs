using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    // I implemented this as an object so I could use it as a reference and "link" neurons between layers.
    // I came to this idea when I saw this implemented in a similar fashion in one of my references:
    // https://www.tech-quantum.com/implement-a-simple-neural-network-in-csharp-net-part-1/
    public class Pulse
    {
        public double Value { get; set; }

        // Some activation functions, like SoftMax, require a raw score (non-activated value).
        // There are a multitude of ways I could have handled that constraint, 
        // but this seemed the easiest at this point.  ;)
        // public double RawScore { get; set; }
    }
}
