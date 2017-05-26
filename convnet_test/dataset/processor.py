import settings
import os.path
from PIL import Image
import numpy as np
import h5py


def image_to_array(file):
    with Image.open(file).convert("RGB") as img:
        img_arr = np.rollaxis(np.array(img), 2, 0)
    return img_arr.tolist()

def get_dataset(data_file, data_path):
    with open(data_file, "rb") as f:
        data = [l.split() for l in f]

    labels = []
    features = []
    
    for line in data:
        feature_file = data_path + line[0]
        
        if (not os.path.isfile(feature_file)):
            print "Error: {} is not a file. Removed from features".format(feature_file)
            continue
        
        feature = image_to_array(feature_file)
        features.append(feature)
        
        label = line[1:]
        if (len(label) <= 1):
            label = label[0]
        labels.append(label)

    return (features, labels)
    

def process(dataset):
    data_path = settings.DATA_PATH + dataset + "/"
    data_file = settings.DATA_PATH + dataset + "/data.txt"
    process_file = settings.PROCESSED_PATH + dataset + ".h5.gz"
    
    if (not os.path.isfile(data_file)):
        print "Error: {} is not a valid dataset.".format(dataset)
        return
        
    features, labels = get_dataset(data_file, data_path)
        
    if not features:
        print "Error: {} dataset is empty.".format(dataset)
        return
    
    processed_data = {"features": features, "labels": labels}
    
    print "Finished parsing dataset."
    
    h5f = h5py.File(process_file, "w")
    for k, v in processed_data.items():
        h5f.create_dataset(k, data=np.array(v), compression="gzip", compression_opts=4)
    h5f.close()
    
    print "Finished writing data."
