import numpy as np

class Student(object):


    """

    This environment simulates learning from exercises. It has the same interface as an OpenAI gym.

    Each student has different strengths and weaknesses, and learns with a different rate.

    Students are processed one at a time, and a new student is retrieved with .reset().

    For each action of step(exercise) an (observation, reward, done, info) tuple is returned, where the observation is: (correct/incorrect bool, time_till_next_exercise integer)

    Each student will do exercises for 2 weeks, then rest for 2 weeks and then perform a final test over each "skill".
    The environment then needs to be .reset().

    The reward for each action is 0, except when 2 weeks has passed where the reward == the test score over all skills.


    The key to achieve good learning in the simulator is to space repetition to
    move memories to long term memory.


    """


    def __init__(self, num_skills=30, random_seed=42):
        self.max_time = 24*14
        self.num_skills = num_skills
        self.action_space = num_skills
        self.observation_space = (bool, int)

        np.random.seed(random_seed)
        self.random_state = np.random.get_state() # To make each student deterministic

        self._reset()

    def _reset(self):
        np.random.set_state(self.random_state)

        self.half_life = 10*np.ones(self.num_skills) # random decay rates, units of 2.4h
        self.ps = np.zeros(self.num_skills) # initial skill skill levels
        self.learn_rate = 1 + np.random.rand(1)  # random learning rate for long term memory
        self.time = 0

        # The following is how different students are "different", i.e. have an easier time learing some skills
        self._initial_exercises()
        self._pass_time(24*28)

        self.random_state = np.random.get_state()


    def _initial_exercises(self, N=100):
        for i in range( N ):
            self._do_exercise(np.random.randint(self.ps.size))
            self.time = 0

    def _pass_time(self, t):

        # time has passed so decay probability:
        # time is measured in hours (could fix this later)
        self.ps *= np.power( 2, -t/2.4/self.half_life ) # Decay probability of recall

    def _do_exercise(self, exercise):
        np.random.set_state(self.random_state)

        correct_prob = self.ps[exercise]

        # reset the skill we just practiced:
        self.ps[exercise] = 1.0

        # decay the decay rate depending on probability:
        self.half_life[exercise] *= np.power( 2, 4*self.learn_rate*correct_prob * (1 - correct_prob) )

        time_till_next = (np.random.rand() < 0.24) * (np.random.poisson(10)) # hours. on average 24h on 10 exercises
        time_till_next = min(self.max_time - self.time, time_till_next)

        self.time += time_till_next

        self._pass_time(time_till_next)

        done = self.time >= self.max_time

        reward = 0.0

        if done:
            reward = self._eval()

        randv = np.random.rand()
        correct = randv < correct_prob

        self.state = np.random.get_state()
        self.random_state = self.state

        return (correct, time_till_next), reward, done

    def _eval(self):
        tmp_ps = self.ps.copy()

        self._pass_time(24*14)
        reward = self.ps.mean()

        self.ps = tmp_ps

        return reward


    def do_exercise(self, exercise):

        if self.time >= self.max_time:
            done = True
            return (0, 0), 0, done

        rv = self._do_exercise(exercise)

        return rv


    def step(self, action):
        return self.do_exercise(action) + ({},)


    def reset(self):
        self._reset()



class StudentEnv(Student):

    def __init__(self):
        super(StudentEnv, self).__init__()

        self.history = np.zeros([10*(self.max_time), self.action_space, 3])
        self.history_length = 1
        self.observation_space = [None, self.action_space, 3]

    def step(self, action):
        (correct, passed_time), reward, done, info = super(StudentEnv, self).step(action)
        self.history[self.history_length, action, 0] = 1
        self.history[self.history_length, action, 1] = correct
        self.history_length += 1

        if passed_time:
            self.history[self.history_length, :, 2] = passed_time
            self.history_length += 1

        return self.history[:self.history_length], reward, done, info

    def reset(self):
        super(StudentEnv, self).reset()

        self.history[:self.history_length] = 0.0

        self.history_length = 1

        return self.history[:self.history_length]



def test():
    s = Student()

    done = False

    while not done:
        a = np.random.randint(0, s.num_skills)
        obs, reward, done = s.do_exercise(a)

    s.reset()




if __name__ == '__main__':
    test()