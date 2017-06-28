# DL-kollo

This environment simulates learning from exercises. It has the same interface as an OpenAI gym.

Each student has different strengths and weaknesses, and learns with a different rate.

Students are processed one at a time, and a new student is retrieved with .reset().

For each action of step(exercise) an (observation, reward, done, info) tuple is returned, where the observation is: (correct/incorrect bool, time_till_next_exercise integer)

Each student will do exercises for 2 weeks, then rest for 2 weeks and then perform a final test over each "skill".
The environment then needs to be .reset().

The reward for each action is 0, except when 2 weeks has passed where the reward == the test score over all skills.


The key to achieve good learning in the simulator is to space repetition to
move memories to long term memory.


