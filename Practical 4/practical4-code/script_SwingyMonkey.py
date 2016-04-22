import numpy as np
import matplotlib.pyplot as plt
import pickle


if __name__ == '__main__':
	# Run online Q-learning (may take ~20 minutes)
	execfile('QMonkey.py')

	# Run Batch RL on SwingyMonkey (may take ~40 minutes, watch and be impressed :) )
	execfile('BatchMonkey.py')

	# Load Results
	with open('random_hist','r') as randf: # Load results from random policy
		random_hist = pickle.load(randf)

	with open('qlearn_online_hist','r') as qf: # Load online Q-learning results
		qlearn_hist = pickle.load(qf)

	with open('batch_hist','r') as bf: # Load batch RL results
		batch_hist = pickle.load(bf)

	# Plot Results
	plt.figure(figsize=(12,8))
	plt.plot(np.cumsum(random_hist['score_history']),lw=2,label='Random policy')
	plt.plot(np.cumsum(qlearn_hist['score_history']),lw=2,label='Learned policy (Epoch iterated Q)')
	plt.plot(np.cumsum(batch_hist['score_history']),lw=2,label='Learned policy (Batch RL)')
	plt.xlabel('Epoch',fontsize=14)
	plt.ylabel('Cumulative Score',fontsize=14)
	plt.title("Cumulative Score of SwingyMonkey by Policy",fontsize=16)
	plt.legend(loc=0,fontsize=10)
	plt.savefig("SwingyMonkey_3policies_100epochs.png")
	plt.show()
	