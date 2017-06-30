import numpy as np

from kollo import env
Student = env.Student

Student = env.Student

s = Student()
s2 = Student()

rewards = []
rewards2 = []

for _ in range(100):

    # print "Result:"
    # print (s._eval(), np.percentile( s.half_life, [5, 50, 95] ), s.half_life.mean() )


    done = False
    while not done:
        obs, reward, done = s.do_exercise(np.random.randint(0, s.num_skills))


    counter = 0
    done = False
    while not done:
        obs, reward2, done = s2.do_exercise(counter % s2.action_space)

    # print "Final:"
    # print (reward, np.percentile( s.half_life, [5, 50, 95] ), s.half_life.mean() )
    s.reset()
    s2.reset()

    rewards.append(reward)
    rewards2.append(reward2)


print("{} +- {}".format(np.mean(rewards), np.std(np.array(rewards) - np.array(rewards2))))
print("{} +- {}".format(np.mean(rewards2), np.std(np.array(rewards) - np.array(rewards2))))
