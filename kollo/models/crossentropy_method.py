import numpy as np

import keras

from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory

from kollo import simulators


np.set_printoptions(precision=2, suppress=True)


se = simulators.StudentEnv()


def model_fn():
    shared_lstm = keras.layers.LSTM(4)
    shared_dense1 = keras.layers.Dense(4, activation='relu')
    shared_dense2 = keras.layers.Dense(1)

    inp = keras.layers.Input([1, None, se.action_space, 3])

    # take out the first (from multiple states) and each action ()
    inputs = [ keras.layers.Lambda(lambda x: x[:, 0, :, i, :])(inp) for i in range(se.action_space) ]

    lstm_out = [shared_lstm(x) for x in inputs]
    dense_out1 = [shared_dense1(x) for x in lstm_out]
    dense_out2 = [shared_dense2(x) for x in dense_out1]

    out = keras.layers.Concatenate()(dense_out2)

    return keras.models.Model(inp, out)



model = model_fn()

print(model.summary())



memory = EpisodeParameterMemory(limit=1000, window_length=1)

cem = CEMAgent(model=model, nb_actions=se.action_space, memory=memory,
               batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
cem.compile()

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
history = cem.fit(se, nb_steps=100000, visualize=False, verbose=2)

# After training is done, we save the best weights.
cem.save_weights('cem_{}_params.h5f'.format('Student2'), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
cem.test(se, nb_episodes=5, visualize=False)
