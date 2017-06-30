import numpy as np

import keras

from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory

from kollo import env


N_FEATURES = 3

class StudentEnv(env.Student):

    def __init__(self):
        super(StudentEnv, self).__init__()

        # self.history = np.zeros([10*(self.max_time), self.action_space, N_FEATURES])
        self.history = [np.zeros([10*(self.max_time), N_FEATURES]) for _ in range(self.action_space)]
        self.history_length = 1

    def step(self, action):
        (correct, passed_time), reward, done, info = super(StudentEnv, self).step(action)
        # self.history[self.history_length, action, 0] = 1
        # self.history[self.history_length, action, 1] = correct
        self.history[action][self.history_length, 0] = 1
        self.history[action][self.history_length, 1] = correct
        self.history_length += 1
        if passed_time:
            # self.history[self.history_length, :, 2] = passed_time
            self.history[action][self.history_length, 2] = passed_time
            self.history_length += 1

        # return self.history[:self.history_length], reward, done, info
        return [x[:self.history_length] for x in self.history], reward, done, info

    def reset(self):
        super(StudentEnv, self).reset()

        # self.history[:self.history_length] = 0.0
        for x in self.history:
            x[:self.history_length] = 0.0

        self.history_length = 1

        # return self.history[:self.history_length]
        return [x[:self.history_length] for x in self.history]



se = StudentEnv()


def model_fn():
    shared_lstm = keras.layers.LSTM(64)
    shared_dense = keras.layers.Dense(1)

    inputs = [keras.layers.Input([None, N_FEATURES], name='input{}'.format(i)) for i in xrange(se.action_space)]
    # inp = keras.layers.Input([None, se.action_space, N_FEATURES])

    # lstm_out = [shared_lstm(inp[:, i]) for i in range(se.action_space)]
    lstm_out = [shared_lstm(x) for x in inputs]
    dense_out = [shared_dense(x) for x in lstm_out]

    out = keras.layers.Concatenate()(dense_out)

    return keras.models.Model(inputs, out)


model = model_fn()

print(model.summary())



memory = EpisodeParameterMemory(limit=1000, window_length=1)

cem = CEMAgent(model=model, nb_actions=se.action_space, memory=memory,
               batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
cem.compile()

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
cem.fit(se, nb_steps=100, visualize=False, verbose=2)

# After training is done, we save the best weights.
cem.save_weights('cem_{}_params.h5f'.format('Student1'), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
cem.test(env, nb_episodes=5, visualize=True)
