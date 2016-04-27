## Introduction

We created two separate scripts to perform reinforcement learning on the SwingyMonkey game. We approached the problem via function approximation under the assumption of a continuous state space.

The first method performs online policy iteration with Q-learning after each epoch, with decreased exploration after every 10 epochs.
After 100 training epochs we take the learned policy and apply it without exploration to 100 epochs of SwingyMonkey to test the policy.

The file that runs this is titled QMonkey.py

The second method uses a batch reinforcement learning approach as part of the policy iteration/ Q-learning.
Initiallly we run 100 epochs based on a randomized policy compile the received data into a batch and train a random forest regressor (using Extra Random Trees) to learn the best policy for that batch.
Then we run 20 intermediate batches using 50 epochs, retraining our policy on the aggregate of all previous data.
*All of these batches are run with a 10 percent exploration rate.*
Finally, we test our optimal policy based on all previously received state,action,reward information without any exploration with a duration of 100 epochs.

The file that runs this batch RL method is titled BatchMonkey.py

## Execution instructions
**We created a script that runs, loads and plots results of our methods titled script_SwingyMonkey.py**

To run both methods together, run python script_SwingyMonkey.py

To run our online q-learning method by itself, run python QMonkey.py

To run the batch reinforcement learning method by itself, run python BatchMonkey.py

 
