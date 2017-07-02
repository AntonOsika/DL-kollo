import numpy as np

import keras
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from kollo import simulators


np.set_printoptions(precision=2, suppress=True)


se = simulators.StudentEnv()


def model_fn():
    shared_lstm = keras.layers.LSTM(4)
    shared_dense1 = keras.layers.Dense(4, activation='relu')

    inp = keras.layers.Input([1, None, se.action_space, 3])

    # take out the first (from multiple states) and each action ()
    inputs = [ keras.layers.Lambda(lambda x: x[:, 0, :, i, :])(inp) for i in range(se.action_space) ]

    lstm_out = [shared_lstm(x) for x in inputs]
    dense_out1 = [shared_dense1(x) for x in lstm_out]

    out = keras.layers.Concatenate()(dense_out1)

    return keras.models.Model(inp, out)



model = model_fn()

# print(model.summary())



memory = SequentialMemory(limit=1000, window_length=1)

policy = BoltzmannQPolicy()

dqn = DQNAgent(model=model, nb_actions=se.action_space, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
history = dqn.fit(se, nb_steps=50000, visualize=False, verbose=2)

rewards = [ x for x in history.history['episode_reward'] if x > 0 ]

import matplotlib.pyplot as plt

plt.plot(np.convolve(np.ones(100), rewards, 'valid'))
plt.show()


# After training is done, we save the best weights.
dqn.save_weights('dqn_{}_params.h5f'.format('Student2'), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(se, nb_episodes=5, visualize=False)
