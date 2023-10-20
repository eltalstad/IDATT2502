import keras
from functions_final import DeepQLearning
import gym
import numpy as np

# load the model
loaded_model = keras.models.load_model("trained_model.h5", custom_objects={'my_loss_fn': DeepQLearning.my_loss_fn})

sumObtainedRewards = 0
# simulate the learned policy for verification


# create the environment, here you need to keep render_mode='rgb_array' since otherwise it will not generate the movie
env = gym.make("CartPole-v1", render_mode='human')
# reset the environment
(currentState, prob) = env.reset()

# since the initial state is not a terminal state, set this flag to false
terminalState = False
while not terminalState:
    # get the Q-value (1 by 2 vector)
    Qvalues = loaded_model.predict(currentState.reshape(1, 4))
    # select the action that gives the max Qvalue
    action = np.random.choice(np.where(Qvalues[0, :] == np.max(Qvalues[0, :]))[0])
    # if you want random actions for comparison
    # action = env.action_space.sample()
    # apply the action
    (currentState, currentReward, terminalState, _, _) = env.step(action)
    # sum the rewards
    sumObtainedRewards += currentReward

env.reset()
env.close()
