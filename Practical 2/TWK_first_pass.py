import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse

import utils
import TWK_feat_eng
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split


TRAIN_DIR = "train"
# TEST_DIR = "test"

def create_submission(ids,predictions,filename):
	with open(filename, "w") as f:
		f.write("Id,Prediction\n")
		for i,p in enumerate(predictions):
			f.write(str(i+1) + "," + str(p) + "\n")

if __name__ == "__main__":
	num_of_train_files = len(os.listdir(TRAIN_DIR))
	X_train, t_train, train_ids = TWK_feat_eng.create_data_matrix(0, num_of_train_files, TRAIN_DIR, training=True)

	xX_train, xX_valid, xY_train, xY_valid = train_test_split(X_train, t_train, test_size=0.33, random_state=181)

	# plt.figure()
	# plt.hist(xY_train,bins=20,alpha=0.5)
	# plt.hist(xY_valid,bins=20,alpha=0.5)
	# plt.show()





