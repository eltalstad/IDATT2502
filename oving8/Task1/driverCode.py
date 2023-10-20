import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

# import the class that implements the Q-Learning algorithm
from functions import Q_Learning

# env=gym.make('CartPole-v1',render_mode='human')
env = gym.make('CartPole-v1')
(state, _) = env.reset()
# env.render()
# env.close()

# here define the parameters for state discretization
upperBounds = env.observation_space.high
lowerBounds = env.observation_space.low
cartVelocityMin = -3
cartVelocityMax = 3
poleAngleVelocityMin = -10
poleAngleVelocityMax = 10
upperBounds[1] = cartVelocityMax
upperBounds[3] = poleAngleVelocityMax
lowerBounds[1] = cartVelocityMin
lowerBounds[3] = poleAngleVelocityMin

numberOfBinsPosition = 30
numberOfBinsVelocity = 30
numberOfBinsAngle = 30
numberOfBinsAngleVelocity = 30
numberOfBins = [numberOfBinsPosition, numberOfBinsVelocity, numberOfBinsAngle, numberOfBinsAngleVelocity]

# define the parameters
alpha = 0.1
gamma = 1
epsilon = 0.2
numberEpisodes = 1000

# create an object
Q1 = Q_Learning(env, alpha, gamma, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds)
# run the Q-Learning algorithm
Q1.simulateEpisodes()
# print the Q-matrix
print("Q-matrix:")
print(Q1.Qmatrix)

# Convert the Q-matrix to a pandas DataFrame
q_matrix_flat = Q1.Qmatrix.reshape(-1, Q1.Qmatrix.shape[-1])
df = pd.DataFrame(q_matrix_flat)

# Add column names
df.columns = [f'Action{i}' for i in range(Q1.actionNumber)]

# Add row names
index_names = []
for i in range(Q1.numberOfBins[0]):
    for j in range(Q1.numberOfBins[1]):
        for k in range(Q1.numberOfBins[2]):
            for l in range(Q1.numberOfBins[3]):
                index_names.append(f'State_{i}_{j}_{k}_{l}')
df.index = index_names

# Save the DataFrame to a CSV file
df.to_csv('q_matrix.csv')
# simulate the learned strategy
(obtainedRewardsOptimal, env1) = Q1.simulateLearnedStrategy()

plt.figure(figsize=(12, 5))
# plot the figure and adjust the plot parameters
plt.plot(Q1.sumRewardsEpisode, color='blue', linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Sum of Rewards in Episode')
plt.yscale('log')
plt.savefig('convergence.png')
plt.show()

# close the environment
env1.close()
# get the sum of rewards
np.sum(obtainedRewardsOptimal)

# now simulate a random strategy
(obtainedRewardsRandom, env2) = Q1.simulateRandomStrategy()
plt.hist(obtainedRewardsRandom)
plt.xlabel('Sum of rewards')
plt.ylabel('Percentage')
plt.savefig('histogram.png')
plt.show()

for i in range(100):
    (obtainedRewardsOptimal, env1) = Q1.simulateLearnedStrategy()