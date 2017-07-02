import numpy as np

from kollo import simulators
Student = simulators.Student

Student = simulators.Student

s = Student()
s2 = Student()
s3 = simulators.SimpleStudent()

rewards = []
rewards2 = []
rewards3 = []

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

    done = False
    while not done:
        obs, reward3, done = s3.do_exercise(np.random.randint(0, s3.num_skills))

    # print "Final:"
    # print (reward, np.percentile( s.half_life, [5, 50, 95] ), s.half_life.mean() )
    s.reset()
    s2.reset()
    s3.reset()

    rewards.append(reward)
    rewards2.append(reward2)
    rewards3.append(reward3)


print("random: {} +- {}".format(np.mean(rewards), np.std(np.array(rewards) - np.array(rewards2))))
print("spread out: {} +- {}".format(np.mean(rewards2), np.std(np.array(rewards) - np.array(rewards2))))
print("random for simple: {} +- {}".format(np.mean(rewards3), np.std(np.array(rewards3))))
