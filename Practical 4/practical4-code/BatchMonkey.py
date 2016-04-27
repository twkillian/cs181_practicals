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

    def random_actions(self,state):
        '''This function is just a place holder to run a random policy
         in order to generate a batch of state, action, reward pairings'''

        new_action = npr.rand() < 0.1
        new_state  = state

        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

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


def run_games(learner, hist, policy='random', eps=0.9, gam=0.5, alph = 0.75, iters = 20, t_len = 100):
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
        
        if policy == 'random':
            swing = SwingyMonkey(sound=False,
                                 text="Random Epoch %d" % (ii),
                                 tick_length = t_len,
                                 action_callback=learner.random_actions,
                                 reward_callback=learner.reward_callback)

        else:
            swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                                 text="Learned Epoch %d" % (ii),       # Display the epoch on screen.
                                 tick_length = t_len,          # Make game ticks super fast.
                                 action_callback=learner.action_callback,
                                 reward_callback=learner.reward_callback)

            learner.fitted = True

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

        # Reset the state of the learner.
        learner.reset()

    hist['state_history'] = hist['state_history'] + total_states
    hist['action_history'] += total_actions
    hist['reward_history'] += total_rewards
    hist['score_history'] += total_scores
        
    return


if __name__ == '__main__':

	# Select agent.
    agent = Learner()

	# Initialize estimator
    extraTrees = ExtraTreesRegressor(n_estimators=50)
    num_batches = 10

    # Empty list to save history.
    hist = {}
    hist['state_history'] = []
    hist['action_history'] = []
    hist['reward_history'] = []
    hist['score_history'] = []

	# Run games with random policy. 
    run_games(agent, hist, policy='random',iters=100, t_len=1)

    # Save random policy iteration as baseline
    with open("random_hist","w") as g:
        pickle.dump(hist ,g)

    # Train fitted q policy based off of random policy
    X_train = np.array([np.append(hist['state_history'][kk],hist['action_history'][kk]) for kk in range(len(hist['state_history']))])
    y_train = np.array(hist['reward_history'])

    extraTrees.fit(X_train, y_train)
    agent.estimator = extraTrees

    # Run a few fitting iterations based on previous game play, refine the estimator
    for i_batch in range(num_batches):

        run_games(agent, hist, policy='q',iters=50,t_len=1)

        X_train = np.array([np.append(hist['state_history'][kk],hist['action_history'][kk]) for kk in range(len(hist['state_history'])-1)])
        # Construct Bellman's equations to get expected rewards based on next proposed state.
        y_train = np.array([agent.estimator.predict(np.append(hist['state_history'][kk],hist['action_history'][kk])) \
            +agent.alph*(hist['reward_history'][kk]+(agent.gam * np.max([agent.estimator.predict(np.append(hist['state_history'][kk+1]\
            ,act)) for act in range(agent.num_actions)]))-agent.estimator.predict(np.append(hist['state_history'][kk],hist['action_history'][kk])))\
             for kk in range(len(hist['state_history'])-1)])

        extraTrees = ExtraTreesRegressor(n_estimators = 50)
        extraTrees.fit(X_train,y_train)
        agent.estimator = extraTrees

    with open("batch_learning_hist","w") as g2:
        pickle.dump(hist,g2)

    # Run game for final 100 iterations
    final_hist = {}
    final_hist['state_history'] = []
    final_hist['action_history'] = []
    final_hist['reward_history'] = []
    final_hist['score_history'] = []

    # Run game based on optimal policy with no exploration!
    run_games(agent,final_hist,policy='q',eps = 1.1, iters=100, t_len=5)
    

	# Save history (maybe only the first and last stages). 
    with open("batch_hist","w") as f:
        pickle.dump(final_hist ,f)
	# np.save('hist',np.array(hist))


