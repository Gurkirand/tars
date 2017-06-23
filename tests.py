from models import RNN_cell
from models import LSTM_cell
import coder
import numpy as np
import tensorflow as tf


TEXT = ["If you knew the algorithm and fed it back say ten thousand times, each time there'd be a dot somewhere on the screen.", " You'd never know where to expect the next dot.", " But gradually you'd start to see this shape, because every dot will be inside the shape of this leaf.", " It wouldn't be a leaf, it would be a mathematical object.", " But yes.", " The unpredictable and the predetermined unfold together to make everything the way it is.", " It's how nature creates itself, on every scale, the snowflake and the snowstorm."]

def test():
    features = []
    labels = []
    for sentence in TEXT:
        data = coder.encode(sentence)
        features.append(np.array(data[:-1]))
        labels.append(np.array(data[1:]))
        
    input_size = output_size = coder.ONE_HOT_SIZE
    batch_size = 1
    epochs = 100
    learning_rate = 0.0001
    
    seed = coder.encode("If you knew the algorithm")
    
    lstm = LSTM_cell(input_size, 120, output_size, model_name="valentine")
    o = lstm.generate(8, np.array([seed]))
    for _o in o:
        print coder.decode(_o)
    # lstm.train(features, labels, epochs, learning_rate)
    
test()

#OLD:
    # data = coder.encode(TEXT)
    
    # data_size = len(TEXT)
    # input_size = output_size = coder.ONE_HOT_SIZE
    # batch_size = 1
    # epochs = 100
    # learning_rate = 0.01
    
    # print data[0:2]
    
    # features = np.array([data[:-1]])
    # labels = np.array([data[1:]])
    
    # print "Features shape: ", features.shape
    # print "Labels shape: ", labels.shape
    
    # rnn = RNN_cell(input_size, 100, output_size)
    # rnn.train(features, labels, batch_size, epochs, learning_rate)
    # o = rnn.run(features[0:6], 20)
    # o = np.array(o).T.tolist()[0]
    # print o
    # print coder.decode(o)
    # for output in labels[0:10]:
    #     print np.argmax(output, axis=-1)
    
