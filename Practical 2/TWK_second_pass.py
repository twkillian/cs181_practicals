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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
	good_attributes = TWK_feat_eng.read_attributes('attributes.txt')
	good_calls = TWK_feat_eng.read_attributes('calls.txt')
	X_train, t_train, train_ids = TWK_feat_eng.create_data_matrix(0, num_of_train_files, good_attributes, good_calls, direc=TRAIN_DIR, training=True)
	full_test, _, test_ids = TWK_feat_eng.create_data_matrix(0, num_of_test_files, good_attributes, good_calls, direc=TEST_DIR, training=False)

	xX_train, xX_valid, xY_train, xY_valid = train_test_split(X_train, t_train, test_size=0.33, random_state=181)

	# Quickly check to see if distributions are well mixed... first pass showed that the histograms matched up well
	# plt.figure()
	# plt.hist(xY_train,bins=20,alpha=0.5)
	# plt.hist(xY_valid,bins=20,alpha=0.5)
	# plt.show()

	print "We've compiled the training data and have split it into training/validation... stay tuned!\n"
	print "Train set dims: ", X_train.shape, "Number of training files: ", num_of_train_files
	print "Test set dims: ", full_test.shape, "Number of testing files: ", num_of_test_files

	# Standardize the data!
	scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
	xX_train = scaler.fit_transform(xX_train)
	xX_valid = scaler.transform(xX_valid)

	X_train = scaler.fit_transform(X_train)
	full_test = scaler.fit(full_test)

	n_folds = 10
	n_jobs = 1

	# Initialize different classifiers
	logReg_clf = LogisticRegression()
	rf_clf = RandomForestClassifier()
	gb_clf = GradientBoostingClassifier()

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

	# Just to finely tune our Random Forest, I'm going to pass it through a larger gridsearch

	rf_params = {"max_depth": [3, None], "max_features": [1, 3, 10], \
	"n_estimators": [50, 100, 200, 300], "min_samples_split": [1, 3, 10], \
	"min_samples_leaf": [1, 3, 10], "bootstrap": [True, False], \
	"criterion": ["gini", "entropy"]}

	gs_rf_clf = GridSearchCV(rf_clf,param_grid=rf_params,cv=n_folds,n_jobs=n_jobs)
	gs_rf_clf.fit(xX_train,xY_train)
	print "BEST RF:   ", gs_rf_clf.best_params_, gs_rf_clf.best_score_
	best_rf_clf = gs_rf_clf.best_estimator_
	best_rf_clf = best_rf_clf.fit(xX_train,xY_train)

	rf_trainingAcc = best_rf_clf.score(xX_train,xY_train)
	rf_validationAcc = best_rf_clf.score(xX_valid,xY_valid)

	print "############### RANDOM FOREST RESULTS ####################"
	print "Accuracy on training data:    %0.2f" % (rf_trainingAcc)
	print "Accuracy on validation data:  %0.2f" % (rf_validationAcc)
	print "#########################################################\n"

	# Now running a gradient boosting classifier, through grid search to finely tune the thing

	gb_params = {"max_depth": [3, None], "max_features": [1, 3, 10], \
	"n_estimators": [10, 25, 50, 100], "min_samples_split": [1, 3, 10], \
	"min_samples_leaf": [1, 3, 10], "learning_rate": [0.001,0.01,0.1,1], \
	"loss": ["deviance", "exponential"], "subsample": [0.25, 0.5, 1., 1.5]}

	gs_gb_clf = GridSearchCV(gb_clf,param_grid=gb_params,cv=n_folds,n_jobs=n_jobs)
	gs_gb_clf.fit(xX_train,xY_train)
	print "Best GB:  " gs_gb_clf.best_params_, gs_gb_clf.best_score_
	best_gb_clf = gs_gb_clf.best_estimator_
	best_gb_clf = best_gb_clf.fiti(xX_train,xY_train)

	gb_trainingAcc = best_gb_clf.score(xX_train,xY_train)
	gb_validationAcc = best_gb_clf.score(xX_valid, xY_valid)

	print "############# GRADIENT BOOSTING RESULTS ##################"
	print "Accuracy on training data:    %0.2f" % (gb_trainingAcc)
	print "Accuracy on validation data:  %0.2f" % (gb_validationAcc)
	print "#########################################################\n"

	# Now predict against full_test set
	print "Now predicting against test set"

	if gb_validationAcc >= rf_validationAcc:
		best_gb_clf.fit(X_train,t_train)
		preds = best_gb_clf.predict(full_test)
	else:
		rf_clf.fit(X_train,t_train)
		preds = rf_clf.predict(full_test)

	# Just out of curiosity, I want to check the histogram of the predictions against the training labels
	plt.figure()
	plt.hist(t_train,bins=20,alpha=0.5)
	plt.hist(preds,bins=20,alpha=0.5)
	plt.show()

	# Now creating submission
	create_submission(test_ids,preds,'third_pass_feat_eng_plusSJDattributes_TWK.csv')











