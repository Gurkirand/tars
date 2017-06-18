from RNN_cell import RNN_cell
import encode_text
import numpy as np
import tensorflow as tf


TEXT = "If you knew the algorithm and fed it back say ten thousand times, each time there'd be a dot somewhere on the screen. You'd never know where to expect the next dot. But gradually you'd start to see this shape, because every dot will be inside the shape of this leaf. It wouldn't be a leaf, it would be a mathematical object. But yes. The unpredictable and the predetermined unfold together to make everything the way it is. It's how nature creates itself, on every scale, the snowflake and the snowstorm."

def test():
    data = encode_text.encode(TEXT)
    
    data_size = len(TEXT)
    input_size = output_size = encode_text.ONE_HOT_SIZE
    batch_size = 50
    epochs = 30
    learning_rate = 0.01
    
    features = data[:-1]
    labels = data[1:]
    
    print "Features shape: ", features.shape
    print "Labels shape: ", labels.shape
    
    rnn = RNN_cell(input_size, 16, output_size)
    rnn.train(features, labels, batch_size, epochs, learning_rate)
    o = rnn.run(features[0], 10)
    print o
    for output in labels[0:10]:
        print np.argmax(output, axis=-1)
    
    
    
    
test()

