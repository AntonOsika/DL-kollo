import numpy as np

class Student:


    """
    This is an environment that simulates learning from exercises.

    Each student has different strengths and weaknesses, and learns with a different rate.

    Students are processed one at a time, and a new student is retrieved with .reset().

    For each action of do_exercise(exercise) an (observation, reward, done) tuple is returned, where the observation is: (correct/incorrect, time_till_next_exercise)

    Each student will do exercises for 30 days, then rests for 7 days and then perform a final test over each "skill".
    The environment (Student) then needs to be .reset().

    The reward for each action is 0, except when 30 days has passed where the reward == the test score over all skills.


    The key to achieve good learning in the simulator is to repeat the exercises, and to take advantage of the free lunch in education: space repetition (to make
    memories get storeed in long term memory).

    
    """

    def __init__(self, num_skills=100):
        self.max_time = 24*30

        self.half_life = 10*np.ones(num_skills) # random decay rates, units of 2.4h 
        self.ps = np.zeros(num_skills) # initial skill skill levels
        self.learn_rate = 1 + np.random.rand(1)  # random learning rate for long term memory
        self.time = 0

        # The following is how different students are "different", i.e. have an easier time learing some skills
        _initial_exercises()
        self._pass_time(7*24) # pass one week

    def _initial_exercises(self, N=300):
        for i in range( N ):
            self._do_exercise(np.random.randint(self.ps.size))[0][0]
            self.time = 0


    def _pass_time(self, t):

        # time has passed so decay probability:
        # time is measured in hours (could fix this later)
        self.ps *= np.power( 2, -t/2.4/self.half_life ) # Decay probability of recalling memory

    def _do_exercise(self, exercise):
        
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

        correct = np.random.rand() < correct_prob

        return (correct, time_till_next), reward, done

    def _eval(self):
        tmp_ps = self.ps.copy()

        self._pass_time(24*7)
        reward = self.ps.sum()

        self.ps = tmp_ps

        return reward


    def do_exercise(self, exercise):

        res = self._do_exercise(exercise)

        return res

    def reset(self):
        self.__init__()

