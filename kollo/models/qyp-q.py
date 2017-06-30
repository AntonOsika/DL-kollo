#!/usr/bin/python2
from __future__ import print_function, division

import time
import random
import gym
import numpy as np
from collections import deque

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


n_episodes = 2000


class DeepQAgent(object):
    def __init__(self,
        state_size,
        action_size,
        memory_size=2000,
        gamma=0.95,
        epsilon_decay=0.995,
        epsilon_min=0.001,
        learning_rate=0.001
    ):
        self.memory = deque(maxlen=memory_size)

        model = Sequential()
        model.add(Dense(24, input_dim=state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=Adam(lr=learning_rate),
        )
        self.model = model

        self.action_size = action_size

        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, observation):
        if np.random.rand() <= self.epsilon:
            random_action = random.randrange(self.action_size)

            return random_action

        else:
            # POLICY : greedy action strategy
            action_quality = self.model.predict(observation)
            action = np.argmax(action_quality[0])

            return action

    def replay(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        # TODO should we make this in a batch and fit the entire batch?
        for state, action, reward, next_state, done in batch:
            q_est = reward
            if not done:
                q_est = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            q_est_ = self.model.predict(state)

            # min L = (Q_(t,theta)(action) - (reward + gamma*max Q_(frozen, t+1)))^2
            # grad Q_(t, theta)(action) *(Q_t,theta(action) - (reward + gamma*max Q_(frozen, t+1)))
            # theta_t+1 = theta_t + grad L_t

            # TODO try if boosting of samples that are way-off on estimation
            # delta_q = abs(q_est - q_est_[0][action])

            q_est_[0][action] = q_est

            self.model.fit(state, q_est_, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self,name):
        self.model.load_weights(name)


    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DeepQAgent(state_size, action_size)

    done = False
    batch_size = 32

    for episode in range(n_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time_step in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
# since the lunarlander gets its reward by landing 
            # reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode : {}/{}, score : {}, e: {:.2}".format(episode,
                                                                     n_episodes,
                                                                     time_step,
                                                                     agent.epsilon))
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

