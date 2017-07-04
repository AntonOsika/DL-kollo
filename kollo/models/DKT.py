import numpy as np

import tensorflow as tf

import keras
import keras.backend as K
from keras.layers.wrappers import TimeDistributed

from kollo import simulators


np.set_printoptions(precision=2, suppress=True)


se = simulators.StudentEnv()


def models_fn():
    shared_lstm = keras.layers.LSTM(4, return_sequences=True)
    shared_dense1 = TimeDistributed(keras.layers.Dense(4, activation='sigmoid'))
    shared_dense2 = TimeDistributed(keras.layers.Dense(1))

    inp_train = keras.layers.Input([None, se.action_space, 3])

    inputs_train = [ keras.layers.Lambda(lambda x: x[:, :, i, :])(inp_train) for i in range(se.action_space) ]

    lstm_out_train = [shared_lstm(x) for x in inputs_train]
    dense_out1_train = [(shared_dense1(x)) for x in lstm_out_train]
    dense_out2_train = [(shared_dense2(x)) for x in dense_out1_train]

    # Mask probabilities that are not used to zero. Takes X and prob.
    mask_and_shift_layer = keras.layers.Lambda(lambda x: x[0][:, 1:, 0:1]*x[1][:, :-1])
    out_train_list = [mask_and_shift_layer([y, z]) for y, z in zip(inputs_train, dense_out2_train)]

    def concatenate_list(x):
        # s = K.shape(x[0])
        # s_list = [s[i] for i in range(s.shape[0].value)]
        # s_list.insert(2, 1)
        # shape = tf.stack(s_list)
        # [tf.reshape(a, shape) for a in x]
        return K.stack(x, axis=2)

    out_train = keras.layers.Lambda(concatenate_list)(out_train_list)


    m_train = keras.models.Model(inp_train, out_train)

    def loss(y, y_pred):
        return K.mean(K.binary_crossentropy(y, y_pred, from_logits=True), axis=-1)

    m_train.compile(loss=loss, optimizer='adam')


    inp_test = keras.layers.Input([None, se.action_space, 3])

    # take out the first (from multiple states) and each action ()
    inputs_test = [ keras.layers.Lambda(lambda x: x[:, :, i, :])(inp_test) for i in range(se.action_space) ]

    lstm_out_test = [shared_lstm(x) for x in inputs_test]
    dense_out1_test = [shared_dense1(x) for x in lstm_out_test]
    dense_out2_test = [shared_dense2(x) for x in dense_out1_test]

    out_test = keras.layers.Lambda(concatenate_list)(dense_out2_test)

    out_test_prob = keras.layers.Activation('sigmoid')(out_test)

    m_test = keras.models.Model(inp_test, out_test_prob)
    m_test.compile(loss='mse', optimizer='sgd')



    return m_train, m_test




def default_dkt(last_embed_dim=200, n_units=4, dropout_input=0.0, dropout_recurrent=0.0, dropout_readout=0.25):

    # Previously had one-hot transform as first layer, commented out:

    # (Rows are selected not columns, this is one-hot transform equivalent)
    # basis = np.concatenate((np.zeros((n_categories+1, 1)), np.eye(n_categories+1)), axis=1).T

    inp = keras.models.Input((None, se.action_space, 3))
    inp_reshaped = keras.layers.Reshape([-1, se.action_space*3])

    # last_embedding = keras.layers.Embedding( 2*n_categories+2, last_embed_dim, mask_zero=True)(inputs[0])

    lstm_output = keras.layers.recurrent.LSTM( n_units, return_sequences=True, recurrent_dropout=dropout_recurrent, dropout=dropout_input )(inp)

    # dot product of next_embedding with LSTM output to get probability. Might be faster with dense layers to categorical crossentropy loss
    next_indices = keras.layers.Lambda(lambda x: x[:, 1:])(inp)

    next_embedding = keras.layers.Embedding( se.action_space, n_units, mask_zero=False)()


    # TODO: Add 1 dim embedding for difficulty of skill. Use dropout on this to create "artificial" data

    #p_hat = keras.layers.dot( [lstm_output, next_embedding], axes=[2, 2] )
    lstm_output_dropout = keras.layers.Dropout(dropout_readout)(lstm_output)

    logit = keras.layers.Lambda( lambda x: K.sum(x[0]*x[1], axis=2) ) ([lstm_output_dropout, inp])

    p_hat = keras.layers.Activation(activation='sigmoid')(logit)

    m = keras.models.Model(inputs=inp, outputs=p_hat)


    m.compile('adam', 'binary_crossentropy', metrics=['acc'])
    return m


# print(model.summary())


n_data = 200
train_data = []


for _ in range(n_data):
    done = False
    while not done:
        obs, reward, done, _ = se.step(np.random.randint(0, se.num_skills))
    train_data.append(obs)
    se.reset()


def data_generator():
    i = 0
    while True:
        yield [ train_data[i%len(train_data)][np.newaxis, :, :, :], train_data[i%len(train_data)][np.newaxis, :-1, :, 1:2] ]
        i += 1

X = [ d[np.newaxis] for d in train_data ]

Y = [ d[np.newaxis, :-1, :, 1:2] for d in train_data ]

train_model, test_model = models_fn()

# history = train_model.fit_generator(data_generator(), steps_per_epoch=n_data, epochs=1)
history = train_model.fit(np.array(X), np.array(Y), epochs=1)


done = False
while not done:
    obs, reward, done, _ = se.step(np.random.randint(0, se.num_skills))
    # print(test_model.predict(obs[np.newaxis]))
    # print(train_model.predict(obs[np.newaxis]))

avg_corr = np.sum(obs[:, :, 0]*obs[:, :, 1])/np.sum(obs[:, :, 0])

avg_corr_pred = np.sum(test_model.predict(obs[np.newaxis])*obs[np.newaxis, :, :, 0:1])/np.sum(obs[:, :, 0])

print("Reward: {} Average corr: {} Pred corr: {} ".format(reward, avg_corr, avg_corr_pred) )
