# DL-kollo

This is an environment that simulates learning from exercises.

Each student has different strengths and weaknesses, and learns with a different rate.

Students are processed one at a time, and a new student is retrieved with .reset().

For each action of do_exercise(exercise) an (observation, reward, done) tuple is returned, where the observation is: (correct/incorrect bool, time_till_next_exercise)

Each student will do exercises for 30 days, then rests for 7 days and then perform a final test over each "skill".
The environment then needs to be .reset().

The reward for each action is 0, except when 30 days has passed where the reward == the test score over all skills.


The key to achieve good learning in the simulator is to repeat the exercises and take advantage of the "free lunch" in education: space repetition (to
move memories to long term memory).


