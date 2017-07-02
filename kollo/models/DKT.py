import numpy as np

import keras

from kollo import simulators


np.set_printoptions(precision=2, suppress=True)


se = simulators.StudentEnv()

n_data = 2000
data = []


def models_fn():
    shared_lstm = keras.layers.LSTM(4)
    shared_dense1 = keras.layers.Dense(4, activation='sigmoid')
    shared_dense2 = keras.layers.Dense(1)

    inp_train = keras.layers.Input([None, 3])


    lstm_out_train = shared_lstm(inp_train)
    dense_out1_train = shared_dense1(lstm_out_train)
    out_train = shared_dense2(dense_out1_train)



    inp_test = keras.layers.Input([None, se.action_space, 3])

    # take out the first (from multiple states) and each action ()
    inputs_test = [ keras.layers.Lambda(lambda x: x[:, :, i, :])(inp_test) for i in range(se.action_space) ]

    lstm_out_test = [shared_lstm(x) for x in inputs_test]
    dense_out1_test = [shared_dense1(x) for x in lstm_out_test]
    dense_out2_test = [shared_dense2(x) for x in dense_out1_test]

    out_test = keras.layers.Concatenate()(dense_out2_test)



    return keras.models.Model(inp_train, out_train), keras.models.Model(inp_test, out_test),




train_model, test_model = models_fn()

# print(model.summary())



for _ in range(n_data):
    corrects = []
    done = False
    while not done:
        obs, reward, done = se.do_exercise(np.random.randint(0, se.num_skills))



cem = CEMAgent(model=model, nb_actions=se.action_space, memory=memory,
               batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
cem.compile()

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
history = cem.fit(se, nb_steps=50000, visualize=False, verbose=2)

rewards = [ x for x in history.history['episode_reward'] if x > 0 ]

import matplotlib.pyplot as plt

plt.plot(np.convolve(np.ones(100), rewards, 'valid'))
plt.show()


# After training is done, we save the best weights.
cem.save_weights('cem_{}_params.h5f'.format('Student2'), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
cem.test(se, nb_episodes=5, visualize=False)
