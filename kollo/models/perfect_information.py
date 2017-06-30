from kollo import env
import  numpy as np
import heapq


Student = env.Student

rewards = np.zeros((20, 100))


s = Student()

# Taking the smallest p:
i = 0
for j in range(100):

    done = False
    while not done:
        action = np.argmin(s.ps)

        obs, rew, done = s.do_exercise(action)

        rewards[i, j] = (rew)

    s.reset()

print("{} +- {}".format(np.mean(rewards[i]), np.std(rewards[i])))


s = Student()

# Taking the action that "minimizes loss in half_life gain (if postponed)" - then the smallest p:
i = 0
for j in range(100):

    done = False
    while not done:
        dPdt = ((1 - 2 *s.ps)* np.log(2)/s.half_life *s.ps)
        action = np.argmax(dPdt)

        if dPdt[action] < 0:
            action = np.argmin(s.ps)

        obs, rew, done = s.do_exercise(action)

        rewards[i, j] = (rew)

    s.reset()

print("{} +- {}".format(np.mean(rewards[i]), np.std(rewards[i])))


s = Student()

# Taking the action that "minimizes loss in half_life gain (if postponed)" - then best improvement for final time :
i = 0
for j in range(100):

    done = False
    while not done:
        dPdt = ((1 - 2 *s.ps)* np.log(2)/s.half_life *s.ps)
        action = np.argmax(dPdt)

        final_p_if_not = s.ps*2**(-(140 + s.max_time - s.time)/s.half_life)
        half_life_if_now = s.half_life * np.power( 2, 4*s.learn_rate*s.ps * (1 - s.ps) )
        final_p_if_now = 2**(-(140 + s.max_time - s.time)/half_life_if_now)

        final_p_improvement_if_now = final_p_if_now - final_p_if_not

        if dPdt[action] < 0 and s.max_time - s.time > 24*2:
            action = np.argmin(s.ps)
        elif dPdt[action] < 0 and s.max_time - s.time < 24 * 2: #tune this
            possible_actions_zipped = sorted(zip(final_p_improvement_if_now, xrange(s.num_skills)), reverse=True)[:max(int((s.max_time-s.time)/2.4)-10, 1)]
            possible_actions = [x for _, x in possible_actions_zipped]

            possible_half_lives = s.half_life[possible_actions]
            action = possible_actions[np.argmax(possible_half_lives)]

        obs, rew, done = s.do_exercise(action)

        rewards[i, j] = (rew)

    s.reset()

print("{} +- {}".format(np.mean(rewards[i]), np.std(rewards[i])))


s = Student()

# Greedily take the action that improves final p the most
i = 0
for j in range(100):

    done = False
    while not done:
        final_p_if_not = s.ps*2**(-(140 + s.max_time - s.time)/s.half_life)
        half_life_if_now = s.half_life * np.power( 2, 4*s.learn_rate*s.ps * (1 - s.ps) )
        final_p_if_now = 2**(-(140 + s.max_time - s.time)/half_life_if_now)

        final_p_improvement_if_now = final_p_if_now - final_p_if_not

        action = np.argmax(final_p_improvement_if_now)

        obs, rew, done = s.do_exercise(action)

        rewards[i, j] = (rew)

    s.reset()

print("{} +- {}".format(np.mean(rewards[i]), np.std(rewards[i])))
