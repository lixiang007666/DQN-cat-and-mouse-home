import pylab
import random
import numpy as np
from environment import Environment
import copy
import tensorflow as tf
from keras.layers import Dense, InputLayer
from tensorflow.python.keras.optimizer_v2.adam import Adam
from keras.models import Sequential

# DeepSARSA Agent for Mouse Cat environment
class Agent:
    def __init__(self):
        self.action_space = [0, 1, 2, 3, 4, 5, 6]
        self.action_size = len(self.action_space) # size of action
        self.state_size = 15 # size of state
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1 # exploration
        self.epsilon_falling = .9999
        self.epsilon_min = 0.01
        self.model = self.build_model()

    # approximate Q function using Neural Network
    # state is input and Q-Value of each action is output of network
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(30, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0]) #maximazing Q-Value

    def train_model(self, state, action, reward, next_state, next_action, end):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon*self.epsilon_falling

        # get maximum Q-Value at s'
        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]
        if end:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor*self.model.predict(next_state)[0][next_action])
        target = np.reshape(target, [1, 7])
        self.model.fit(state, target, epochs = 1, verbose = 0)
        # minibatch which includes target Q-Value

if __name__ == "__main__":
    environment = Environment()
    agent = Agent()
    step = 0
    epsilon = []
    walk = []
    scores = []
    episodes = []

    for e in range(1000):
        end = False
        score = 0
        state = environment.reset()
        state = np.reshape(state, [1, 15])
        previous = step

        # get action for the current state and go one step in environment
        while not end:
            step = step + 1
            action = agent.get_action(state)
            next_state, reward, end = environment.step(action)
            next_state = np.reshape(next_state, [1, 15])
            next_action = agent.get_action(next_state)
            agent.train_model(state, action, reward, next_state, next_action, end)
            state = next_state
            score = score + reward
            state = copy.deepcopy(next_state) #copy for the next episode

            if end:
                print("episode: ", e, "     score: ", score, "     global_step: ", step, "      epsilon: ", agent.epsilon)
                scores.append(score)
                episodes.append(e)
                epsilon.append(agent.epsilon)
                walk.append(step - previous)

# print the final results
pylab.plot(episodes,scores, 'r')
pylab.xlabel('episodes')
pylab.ylabel('scores')
pylab.savefig("./graph_score.png")

pylab.plot(episodes, epsilon, 'r')
pylab.xlabel('episodes')
pylab.ylabel('epsilon')
pylab.savefig("./graph_epsilon.png")

pylab.plot(episodes, walk, 'r')
pylab.xlabel('pisodes')
pylab.ylabel('walk')
pylab.savefig("./graph_step.png")
