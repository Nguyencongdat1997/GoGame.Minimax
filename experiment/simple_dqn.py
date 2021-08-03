import gym
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam


class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_counter = 0

        self.states = np.zeros((self.mem_size, *input_shape), dtype=np.float64)
        self.next_states = np.zeros((self.mem_size, *input_shape), dtype=np.float64)
        self.rewards = np.zeros(self.mem_size, dtype=np.float64)
        self.actions = np.zeros(self.mem_size, dtype=np.int32)
        self.done = np.zeros(self.mem_size, dtype=np.bool)

    def store_step(self, state, action, reward, next_state, done):
        index = self.mem_counter % self.mem_size
        self.states[index] = state
        self.next_states[index] = next_state
        self.actions[index] = action
        self.rewards[index] = reward
        self.done[index] = done
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.states[batch]
        next_states = self.next_states[batch]
        rewards = self.rewards[batch]
        actions = self.actions[batch]
        done = self.done[batch]

        return states, actions, rewards, next_states, done


class DeepQNetwork(keras.Model):
    def __init__(self, n_actions):
        super(DeepQNetwork, self).__init__()

        fc1_dims = 128
        fc2_dims = 128
        self.dense1 = keras.layers.Dense(fc1_dims, activation='relu')
        self.dense2 = keras.layers.Dense(fc2_dims, activation='relu')
        self.Q = keras.layers.Dense(n_actions, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        Q = self.Q(x)
        return Q


class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=1e-3, epsilon_end=0.01,
                 mem_size=1000000, replace=100):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.replace = replace
        self.batch_size = batch_size

        self.learned_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q = DeepQNetwork(n_actions)

        self.q.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    def store_step(self, state, action, reward, next_state, done):
        self.memory.store_step(state, action, reward, next_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
        return action

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        # get data
        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)
        q_pred = self.q(states)
        q_next = self.q(next_states)
        q_target = q_pred.numpy()
        max_next_actions = tf.math.argmax(q_next, axis=1)
        for i, terminated in enumerate(dones):
            q_target[i, actions[i]] = rewards[i] + self.gamma * q_next[i, max_next_actions[i]] * (1 - int(dones[i]))

        # train
        self.q.train_on_batch(states, q_target)

        self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_end)
        self.learned_step_counter += 1

    def train(self, env, n_games):
        scores = []
        eps_history = []
        steps = 0
        for i in range(n_games):
            done = False
            score = 0
            observation = env.reset()
            while not done:
                steps += 1
                action = self.choose_action(observation)
                next_observation, reward, done, info = env.step(action)
                score += reward
                self.store_step(observation, action, reward, next_observation, done)
                observation = next_observation
                self.learn()
            eps_history.append(self.epsilon)
            scores.append(score)
            avg_score = np.mean(scores[-10:])
            print('Episode', i, '- trained steps', steps, '- score %.1f' % score, '- avg_score %.1f ' % avg_score)

    def save_model(self, train_dir):
        file_name = train_dir + '/dqn_' + str(self.learned_step_counter) + '/model'
        self.q.save_weights(file_name, save_format='tf')

    def load_model(self, train_dir, learned_steps=100):
        file_name = train_dir + '/dqn_' + str(learned_steps) + '/model'
        self.q.load_weights(file_name)


# Train
env = gym.make('CartPole-v0')

dqn = Agent(lr=0.005, gamma=0.99, n_actions=env.action_space.n, epsilon=1.0, batch_size=64,
            input_dims=env.observation_space.shape)

n_games = 100
dqn.train(env, n_games)

# Save
# train_dir = '.'
# dqn.epsilon = 0.0
# dqn.save_model(train_dir)
