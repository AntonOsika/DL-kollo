import itertools

import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as layers


import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule


# Requires openAI baselines and python3
# Does not seem to work.
# Does DQN require batch processing?


from kollo import env

s = env.Student()
student_history = [2*s.action_space] # Special character for "beginning of history"

def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = tf.one_hot(out, 2*num_actions + 1)

        cell = tf.nn.rnn_cell.BasicLSTMCell(64)
        rnn_out, last_state = tf.nn.dynamic_rnn(cell, out, dtype=np.float32)
        out = rnn_out[:, -1]

        #out = layers.fully_connected(out, num_outputs=16, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


if __name__ == '__main__':
    np.random.seed(7)

    batch_size = 1 # Necessary, Different lengths
    log_session = logger.session(dir='logs')

    with U.make_session(8):
        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: tf.placeholder(tf.int32, [None, None], name=name),
            q_func=model,
            num_actions=s.action_space,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )
        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        for t in itertools.count(start=1):
            # Take action and update exploration to the newest value
            action = act(np.array(student_history)[None], update_eps=exploration.value(t))[0] #FIXME: shape (0, ) instead of (None, None)

            (correct, time_passed), reward, done = s.do_exercise(action)

            student_history += [ action + correct*s.action_space ] # append observation. Observations are index of exercise + NUM_EXERCISES if it was correct

            # Store transition in the replay buffer.
            replay_buffer.add(np.array(student_history[:-1]), action, reward, np.array(student_history), float(done))

            episode_rewards[-1] += reward
            if done:
                s.reset()
                student_history = student_history[:1]
                episode_rewards.append(0)

            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()

            if done and len(episode_rewards) % 10 == 0:
                    with log_session:
                        logger.record_tabular("steps", t)
                        logger.record_tabular("episodes", len(episode_rewards))
                        logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 3))
                        logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                        logger.dump_tabular()


print("finished")
