from kollo import env
import  numpy as np
import heapq


Student = env.Student

rewards = np.zeros((20, 100))

inc_estimate_factor = 1.8
double_half_life = 3.0
initial1 = 115
initial2 = 35
time_delta = 4.8
# for i, time_delta in enumerate(np.linspace(4, 5.5, 20)):
for i in [0]:
    s = Student()

    for j in range(100):

        heap = [] # stores [ time, action, half_life, var]
        next_exercise = 0

        #print "Starting: {}".format(s._eval())


        done = False
        while not done:
            t0 = s.time

            if (len(heap) > 0 and heap[0][0] - s.time < time_delta ) or next_exercise >= s.num_skills:
                item = heapq.heappop(heap)
                action = item[1]
                obs, rew, done = s.do_exercise( action )

                item[2] *= double_half_life # double half_life
                if obs[0]:
                    item[2] *= inc_estimate_factor # tune this parameter?

                item[0] = t0 + item[2] # add half_life

                heapq.heappush(heap, item)
            else:
                action = next_exercise
                obs, rew, done = s.do_exercise( action )
                if obs[0]:
                    half_life = initial1
                else:
                    half_life = initial2

                next_time = t0 + half_life

                heapq.heappush(heap, [next_time, action, half_life])

                next_exercise += 1

        #print "Done {}".format(rew)

        rewards[i, j] = (rew)

        s.reset()

    print "{}: {} +- {}".format(time_delta, np.mean(rewards[i]), np.std(rewards[i] - rewards[0]))
