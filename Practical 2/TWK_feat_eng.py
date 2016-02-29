# Example Feature Extraction from XML Files
# We count the number of specific system calls made by the programs, and use
# these as our features.

# This code requires that the unzipped training set is in a folder called "train". 

import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse

import utils

TRAIN_DIR = "train"
TEST_DIR = "test"

call_set = set([])

def add_to_set(tree):
    for el in tree.iter():
        call = el.tag
        call_set.add(call)

def read_attributes(filename):
    fp = open(filename,'r')
    return [line.strip() for line in fp]

def create_data_matrix(start_index, end_index, good_attributes, good_calls,direc="train", training=True):
    X = None
    classes = []
    ids = [] 
    i = -1
    for datafile in os.listdir(direc):
        if datafile == '.DS_Store':
            continue

        i += 1
        if i < start_index:
            continue 
        if i >= end_index:
            break

        # extract id and true class (if available) from filename
        id_str, clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(utils.malware_classes.index(clazz))

        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)

        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        if training:
            add_to_set(tree)

    # i = -1
    # for datafile in os.listdir(direc):
        # if datafile == '.DS_Store':
            # continue

        # i += 1
        # if i < start_index:
            # continue 
        # if i >= end_index:
            # break
        this_row = call_feats(tree,good_attributes,good_calls)
        if X is None:
            X = this_row 
        else:
            X = np.vstack((X, this_row))

    return X, np.array(classes), ids

def call_feats(tree, good_attributes, good_calls=None):
    good_calls = ['sleep', 'dump_line', 'impersonate_user','revert_to_self','kill_process','query_value','load','get_computer_name','get_system_directory','query_value','open_key','vm_protect']
    # good_calls = list(good_calls)

    call_counter = {}
    att_counter = {}
    for el in tree.iter():
        call = el.tag

        if call not in call_counter:
            call_counter[call] = 1
        else:
            call_counter[call] += 1
        for att in el.attrib:
            if att in good_attributes:
                if att not in att_counter:
                    att_counter[att] = 1
                else:
                    att_counter[att] +=1

    feat_array = np.zeros(len(good_calls)+len(good_attributes)+1)
    for i in xrange(len(good_calls)):
        call = good_calls[i]
        feat_array[i] = 0
        if call in call_counter:
            feat_array[i] = call_counter[call]

    for i in xrange(len(good_calls),len(good_calls)+len(good_attributes)):
        att = good_attributes[i-len(good_calls)]
        feat_array[i] = 0
        if att in att_counter:
            feat_array[i] = att_counter[att]

    feat_array[len(good_calls)+len(good_attributes)-1] = len(call_counter.keys())+len(att_counter.keys())

    return feat_array

## Feature extraction
def main():
    num_of_train_files = len(os.listdir(TRAIN_DIR))
    num_of_test_files = len(os.listdir(TEST_DIR))

    # Read in attribute names from file
    good_attributes = read_attributes('attributes.txt')
    # good_calls = read_attributes('calls.txt')

    # X_train, t_train, train_ids = create_data_matrix(0, num_of_train_files, good_attributes, good_calls,TRAIN_DIR, training=True)
    # X_test, t_test, test_ids  = create_data_matrix(0, num_of_test_files, good_attributes, good_calls,TEST_DIR, training=False)
    X_train, t_train, train_ids = create_data_matrix(0, num_of_train_files, good_attributes, good_calls,TRAIN_DIR, training=True)
    X_test, t_test, test_ids  = create_data_matrix(0, num_of_test_files, good_attributes, good_calls,TEST_DIR, training=False)

    #
    calls = set(calls_train) | set(calls_test)
    fp = open("calls.txt",'w')
    for call in calls:
        fp.write(call + '\n')
    fp.close()

    print len(call_set)
    print X_train.shape, t_train.shape
    print X_test.shape, t_test.shape

    # From here, you can train models (eg by importing sklearn and inputting X_train, t_train).

if __name__ == "__main__":
    main()
    