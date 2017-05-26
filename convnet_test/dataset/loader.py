import settings
import os.path
import processor
import h5py
import numpy as np
import io
import multiprocessing

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def _load_data(file):
    f = h5py.File(file, "r")
    features = f["features"][:]
    labels = f["labels"][:]
    f.close()
    return (features, labels)

def _load_data_wrapper(file, vectorized):
    tr_d, va_d, te_d = _load_data(file)
    # training_features = tr_d[0]
    # training_labels = tr_d[1]
    # training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    # training_results = [vectorized_result(y) for y in tr_d[1]]
    # training_data = zip(training_inputs, training_results)
    # validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    # validation_data = zip(validation_inputs, va_d[1])
    # test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    # test_data = zip(test_inputs, te_d[1])
    # return (training_data, validation_data, test_data)

def load(dataset, validation=0, train=0, vectorized=False):
    process_file = settings.PROCESSED_PATH + dataset + ".h5.gz"
    _data_file = settings.DATA_PATH + dataset + "/data.txt"
    
    if (os.path.isfile(process_file)):
        return _load_data_wrapper(process_file, vectorized=vectorized)
    elif (os.path.isfile(_data_file)):
        print "Error: {} needs to be processed.".format(dataset)
        return None
        
    print "Error: {} is not a valid dataset.".format(dataset)
    return None
