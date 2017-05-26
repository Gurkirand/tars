import processor
import loader
import time
import json
import gzip
import io
import cPickle as pickle
import numpy as np
import multiprocessing
import time
import h5py

# processor.process("robot")
# start = time.clock()
# loader.load("robot")
# end = time.clock()
# print end - start

# start = time.clock()
# robot_data = processor.get_dataset("raw/robot/data.txt", "raw/robot/")
# end = time.clock()
# print "robot direct read: ", end - start
`
# _data = {}
# _data["f"] = robot_data[0]
# _data["l"] = robot_data[1]

def write(data):
    start = time.clock()
    h5f = h5py.File("processed/mnist.h5.gz", "w")
    for k, v in data.items():
        h5f.create_dataset(k, data=np.array(v), compression="gzip", compression_opts=4)
    # h5f.create_dataset("robot_data", data=np.array(robot_data))
    h5f.close()
    end = time.clock()
    print "robot write hdf5: ", end - start
    
start = time.clock()
h5f = h5py.File("processed/robot.h5.gz", "r")
f_data = h5f["features"][:]
l_data = h5f["labels"][:]
h5f.close()
end = time.clock()
print "mnist read hdf5: ", end - start

# start = time.clock()
# h5f = h5py.File("processed/robot.h5", "r")
# f_data = h5f["f"][:]
# l_data = h5f["l"][:]
# h5f.close()
# end = time.clock()
# print "robot write hdf5: ", end - start

# write(_data)
# p = multiprocessing.Process(target=write, name="Write", args=(_data,))
# p.start()

# p.terminate()
# p.join(100)

# # If thread is active
# if p.is_alive():
#     print "foo is running... let's kill it..."

#     # Terminate foo
#     p.terminate()
#     p.join()
# else:


"""
mnist read gzp:  1.523683
mnist write gzp:  24.175798
mnist2 read gzp:  1.289712
mnist write pkl:  0.467011
mnist read pkl:  0.298679
robot direct read:  5.856837
robot write gzip:  47.937162
robot read gzip:  100.21108
robot write pkl:  16.991457
robot read pkl:  32.076552
robot write json:  155.028894
robot read json:  16.552012
"""

# start = time.clock()
# f = io.BufferedReader(gzip.open("processed/mnist.pkl.gz", "rb"))
# mnist_data = pickle.load(f)
# f.close()
# end = time.clock()
# print "mnist read gzp: ", end - start

# training_data, validation_data, test_data = mnist_data

# mnist_dict = {"training_features": np.array(training_data[0]), "training_labels": np.array(training_data[1]), "validation_features": np.array(validation_data[0]), "validation_labels": np.array(validation_data[1]), "test_features": np.array(test_data[0]), "test_labels": np.array(test_data[1])}
# for k, v in mnist_dict.items():
#     print k
#     print v.shape

# mnist_dict2 = {}
# mnist_dict2["features"] = np.concatenate((mnist_dict["training_features"], mnist_dict["validation_features"], mnist_dict["test_features"]))
# mnist_dict2["labels"] = np.concatenate((mnist_dict["training_labels"], mnist_dict["validation_labels"], mnist_dict["test_labels"])).shape

# write(mnist_dict2)

# start = time.clock()
# f = gzip.open("processed/mnist2.pkl.gz", "wb")
# pickle.dump(mnist_data, f, protocol=pickle.HIGHEST_PROTOCOL)
# f.close()
# end = time.clock()
# print "mnist write gzp: ", end - start

# start = time.clock()
# f = io.BufferedReader(gzip.open("processed/mnist2.pkl.gz", "rb"))
# pickle.load(f)
# f.close()
# end = time.clock()
# print "mnist2 read gzp: ", end - start

# start = time.clock()
# f = open("processed/mnist2.pkl", "wb")
# pickle.dump(mnist_data, f, protocol=pickle.HIGHEST_PROTOCOL)
# f.close()
# end = time.clock()
# print "mnist write pkl: ", end - start

# start = time.clock()
# f = open("processed/mnist2.pkl", "rb")
# pickle.load(f)
# f.close()
# end = time.clock()
# print "mnist read pkl: ", end - start




"""
# faster by a bit
# robot write gzip:  41.083516
# robot read gzip:  90.790373
# robot write pkl:  15.833995
# robot read pkl:  28.785501
# robot write hdf5:  4.977955
# robot write hdf5:  0.515558
# """

# start = time.clock()
# f = gzip.open("processed/robot.pkl.gz", "wb")
# pickle.dump(robot_data, f, protocol=pickle.HIGHEST_PROTOCOL)
# f.close()
# end = time.clock()
# print "robot write gzip: ", end - start

# start = time.clock()
# f = io.BufferedReader(gzip.open("processed/robot.pkl.gz", "rb"))
# pickle.load(f)
# f.close()
# end = time.clock()
# print "robot read gzip: ", end - start

# start = time.clock()
# f = open("processed/robot.pkl", "wb")
# pickle.dump(robot_data, f, protocol=pickle.HIGHEST_PROTOCOL)
# f.close()
# end = time.clock()
# print "robot write pkl: ", end - start

# start = time.clock()
# f = open("processed/robot.pkl", "rb")
# pickle.load(f)
# f.close()
# end = time.clock()
# print "robot read pkl: ", end - start

# start = time.clock()
# f = open("processed/robot.json", "wb")
# json.dump(robot_data, f)
# f.close()
# end = time.clock()
# print "robot write json: ", end - start

# start = time.clock()
# f = open("processed/robot.json", "rb")
# json.load(f)
# f.close()
# end = time.clock()
# print "robot read json: ", end - start

