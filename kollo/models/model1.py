import numpy as np

from kollo import env
Student = env.Student

Student = env.Student

s = Student()

for _ in range(100):

    print "Result:"
    print (s._eval(), np.percentile( s.half_life, [5, 50, 95] ), s.half_life.mean() )


    done = False
    while not done:
        obs, reward, done = s.do_exercise(np.random.randint(0, s.num_skills))

    print "Final:"
    print (reward, np.percentile( s.half_life, [5, 50, 95] ), s.half_life.mean() )
    s.reset()


