# Imports.
import numpy as np
import numpy.random as npr
import pickle
from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        new_action = npr.rand() < 0.1
        new_state  = state

        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey()

        # Initialize history dictionaries for iteration ii
        hist['state'][ii] = []
        hist['action'][ii] = []
        hist['reward'][ii] = []

        # Loop until you hit something.
        while swing.game_loop():

            # This is where we build sarsa arrays utilizing learner.method()
            # You can get the action via learner.last_action (False=0/glide, True=1/jump)
            # You can get the state via learner.last_state
            # You can get the reward via learner.last_reward (0,+1 if pass, -5 if hit, -10 if fall off screen)
            # Can infer gravity by checking monkey velocity from time step to time step if action is false
                # Gravity is an integer 1, 2, 3, or 4

            # import pdb
            # pdb.set_trace()

            hist['state'][ii].append(learner.last_state)
            hist['action'][ii].append(learner.last_action)
            hist['reward'][ii].append(learner.last_reward)

        else: # Get final action,reward and state just to see how the monkey failed.
            hist['state'][ii].append(learner.last_state)
            hist['action'][ii].append(learner.last_action)
            hist['reward'][ii].append(learner.last_reward)
        
        # Save score history.
        hist['score'].append(swing.score)

        # Reset the state of the learner.
        learner.reset()
        
    return


if __name__ == '__main__':

	# Select agent.
    agent = Learner()

	# Empty list to save history.
    hist = {}
    hist['state'] = {}
    hist['action'] = {}
    hist['reward'] = {}
    hist['score'] = []

	# Run games. 
    run_games(agent, hist, 10, 10)

	# Save history. 
    with open("human_hist","w") as f:
        pickle.dump(hist,f)
	# np.save('hist',np.array(hist))


