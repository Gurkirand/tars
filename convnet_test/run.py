from dataset import loader
import NN

training_data, validation_data, test_data = loader.load()
input = len(data[0])
nn = NN.Network([input, 100, 100, 10])
# nn.SGD(training_data, 30, 10, 3.0, test_data=test_data)
