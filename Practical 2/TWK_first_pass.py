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

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV


TRAIN_DIR = "train"
TEST_DIR = "test"

def create_submission(ids,predictions,filename):
	with open(filename, "w") as f:
		f.write("Id,Prediction\n")
		for i,p in zip(ids,predictions):
			f.write(str(i) + "," + str(p) + "\n")

if __name__ == "__main__":
	num_of_train_files = len(os.listdir(TRAIN_DIR))
	num_of_test_files = len(os.listdir(TEST_DIR))
	X_train, t_train, train_ids = TWK_feat_eng.create_data_matrix(0, num_of_train_files, TRAIN_DIR, training=True)
	full_test, _, test_ids = TWK_feat_eng.create_data_matrix(0, num_of_test_files, TEST_DIR, training=False)

	xX_train, xX_valid, xY_train, xY_valid = train_test_split(X_train, t_train, test_size=0.33, random_state=181)

	# Quickly check to see if distributions are well mixed... first pass showed that the histograms matched up well
	# plt.figure()
	# plt.hist(xY_train,bins=20,alpha=0.5)
	# plt.hist(xY_valid,bins=20,alpha=0.5)
	# plt.show()

	print "We've compiled the training data and have split it into training/validation... stay tuned!\n"
	print full_test.shape

	# Standardize the data!
	# scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
	# xX_train = scaler.fit_transform(xX_train)
	# xX_valid = scaler.transform(xX_valid)

	print np.sum(xY_train==8)/float(len(xY_train))
	print np.sum(xY_valid==8)/float(len(xY_valid))

	n_folds = 5
	n_jobs = 1

	# Initialize different classifiers
	logReg_clf = LogisticRegression()
	nb_clf = GaussianNB()
	rf_clf = RandomForestClassifier(n_estimators=50, n_jobs=n_jobs)

	# Pass logistic regression through GridSearchCV, just cause
	Cs=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
	parameters = {"C": Cs}

	gs_logReg_clf = GridSearchCV(logReg_clf, param_grid=parameters, cv=n_folds, n_jobs=n_jobs)
	gs_logReg_clf.fit(xX_train,xY_train)
	print "BEST", gs_logReg_clf.best_params_, gs_logReg_clf.best_score_, gs_logReg_clf.grid_scores_
	best_logReg_clf = gs_logReg_clf.best_estimator_
	best_logReg_clf = best_logReg_clf.fit(xX_train,xY_train)

	logReg_trainingAcc = best_logReg_clf.score(xX_train,xY_train)
	logReg_validationAcc = best_logReg_clf.score(xX_valid,xY_valid)

	print "############# LOGISTIC REGRESSION RESULTS################"
	print "Accuracy on training data:    %0.2f" % (logReg_trainingAcc)
	print "Accuracy on validation data:  %0.2f" % (logReg_validationAcc)
	print "#########################################################\n"

	# Run a simple NaiveBayes classifier
	nb_clf.fit(xX_train,xY_train)

	nb_trainingAcc = nb_clf.score(xX_train,xY_train)
	nb_validationAcc = nb_clf.score(xX_valid,xY_valid)

	print "################# NAIVE BAYES RESULTS####################"
	print "Accuracy on training data:    %0.2f" % (nb_trainingAcc)
	print "Accuracy on validation data:  %0.2f" % (nb_validationAcc)
	print "#########################################################\n"

	rf_clf.fit(xX_train,xY_train)

	rf_trainingAcc = rf_clf.score(xX_train,xY_train)
	rf_validationAcc = rf_clf.score(xX_valid,xY_valid)

	print "############### RANDOM FOREST RESULTS ####################"
	print "Accuracy on training data:    %0.2f" % (rf_trainingAcc)
	print "Accuracy on validation data:  %0.2f" % (rf_validationAcc)
	print "#########################################################\n"

	# Now predict against full_test set
	print "Now predicting against test set"

	if logReg_validationAcc >= rf_validationAcc:
		best_logReg_clf.fit(X_train,t_train)
		preds = best_logReg_clf.predict(full_test)
	else:
		rf_clf.fit(X_train,t_train)
		preds = rf_clf.predict(full_test)

	# Just out of curiosity, I want to check the histogram of the predictions against the training labels
	plt.figure()
	plt.hist(t_train,bins=20,alpha=0.5)
	plt.hist(preds,bins=20,alpha=0.5)
	plt.show()

	# Now creating submission
	create_submission(test_ids,preds,'first_pass_naive_feat_eng_TWK.csv')











