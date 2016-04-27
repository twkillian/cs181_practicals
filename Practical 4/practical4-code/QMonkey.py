# Imports.
import numpy as np
import numpy.random as npr
import pickle
from sklearn.ensemble import ExtraTreesRegressor
from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.eps = None
        self.gam = None
        self.alph = None
        self.estimator = None
        self.num_actions = None
        self.gravity = 4
        self.learn_g = True  # Flag that will be switched off once we've learned g
        self.fitted = False  # Flag that tells us whether we've fit our ExtraTrees

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.gravity = 4
        self.learn_g = True

    def infer_g(self,states,actions):
        '''Here we'll take the last two states after glide actions,
        and use their velocity to infer the gravity term'''
        if np.sum(actions[-2:]) == 0:
            # pulls the monkey's velocity after concurrent glide actions
            g = states[-2][3]-states[-1][3]
            self.gravity = g
            self.learn_g = False

    def create_state_tuple(self, state):
        '''Creates a tuple from the state dictionary provided by SwingyMonkey
        v1.0 utilizes every measure within the game besides the score.
        Future versions may create some features (a la, how far from the bottom we are, etc.)'''

        return np.array(state['tree'].values()+state['monkey'].values()+[self.gravity])

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        # Create tuple from state dictionary, facilitates use in random forest
        st_tuple = self.create_state_tuple(state)

        if not self.fitted:
            new_action = npr.rand() < 0.1
        else:
            # gather new_action in an epsilon greedy manner according to Q estimator
            if npr.rand() > self.eps:
                new_action = npr.rand() < 0.1 # Currently defaults to gliding... may want to adjust
            else:
                new_action = np.argmax([self.estimator.predict(np.append(st_tuple,a)) for a in range(self.num_actions)])
        
        # Store new_state and new_action to pass back to SwingyMonkey
        new_state  = state

        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, eps=0.5, gam=0.5, alph=0.75, iters = 20, t_len = 100, test=False):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''

    # Place alpha and epsilon values into learner
    learner.eps = eps
    learner.gam = gam
    learner.alph = alph
    learner.num_actions = 2

    # Initialize estimator for Q-function

    total_states = []
    total_actions = []
    total_rewards = []
    total_scores = []

    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Initialize history dictionaries for iteration ii
        states = []
        actions = []
        rewards = []
        loop_counter = 0

        # Loop until you hit something.
        while swing.game_loop():

            states.append(learner.create_state_tuple(learner.last_state))
            actions.append(int(learner.last_action==True))
            rewards.append(learner.last_reward)

            if learner.learn_g & (loop_counter > 1):
                learner.infer_g(states,actions)
                for pp in range(len(states)):
                    states[pp][-1] = learner.gravity

            loop_counter += 1

        else: # Get final action,reward and state just to see how the monkey failed.
            states.append(learner.create_state_tuple(learner.last_state))
            actions.append(int(learner.last_action==True))
            rewards.append(learner.last_reward)
        
        # Append histories from most recent epoch, create training arrays
        total_scores.append(swing.score)
        total_states += states
        total_actions += actions
        total_rewards += rewards
        
        if not test:

            # Iteratively refine the optimal policy after each epoch
            if ii == 0:
                X_train = np.array([np.append(total_states[kk],total_actions[kk]) for kk in range(len(total_states))])
                y_train = np.array(total_rewards)

                #Build tree using first stage Q-learning
                extraTrees = ExtraTreesRegressor(n_estimators=50)
                extraTrees.fit(X_train, y_train)

            # Refit random forest estimator based on composite epochs
            
            else:
                # Generate new X(state,action) and y(reward) lists from newly run batch, based off of Q-estimator and using prior rewards a la Ernst '06'
                X_train = np.array([np.append(total_states[kk],total_actions[kk]) for kk in range(len(total_rewards)-1)])
                # Construct Bellman's equations to get expected rewards based on next proposed state
                y_train = np.array([agent.estimator.predict(np.append(total_states[kk],total_actions[kk])) \
                    +agent.alph*(total_rewards[kk]+(agent.gam * np.max([agent.estimator.predict(np.append(total_states[kk+1]\
                    ,act)) for act in range(agent.num_actions)]))-agent.estimator.predict(np.append(total_states[kk],total_actions[kk])))\
                    for kk in range(len(total_states)-1)])
                
                # Re-fit regression to refine optimal policy according to expected reward.
                extraTrees = ExtraTreesRegressor(n_estimators=50)
                extraTrees.fit(X_train,y_train)

            # As we refine the policy, we should reduce the amount we explore.    
            if ii % 10 == 0:
                learner.eps += 0.05

            learner.estimator = extraTrees
            learner.fitted = True

        else:

            learner.fitted = True        

        # Reset the state of the learner.
        learner.reset()

    # Place state, action, reward and score histories to be saved by wrapper.    
    hist['state_history'] = total_states
    hist['action_history'] = total_actions
    hist['reward_history'] = total_rewards
    hist['score_history'] = total_scores 
    return


if __name__ == '__main__':

	# Select agent.
    agent = Learner()

	# Empty list to save history.
    hist = {}

	# Run games. 
    run_games(agent, hist, iters=100, t_len=2)

    with open("qlearn_online_hist","w") as g:
        pickle.dump(hist,g)

    # With optimal policy, run test
    final_hist = {}
    run_games(agent,final_hist,iters=100,t_len=5,test=True,eps=1.1)

	# Save history. 
    with open("qlearn_final_hist","w") as f:
        pickle.dump(final_hist,f)
	# np.save('hist',np.array(hist))


