from kollo import env
Student = env.Student

s1 = env.Student()
s2 = env.Student()

s = ""

done = False
while not done:
    obs, reward, done = s1.do_exercise(0)
    s += str(obs[1])

print "Should be identical:"
print s

s = ""
done = False
while not done:
    obs, reward, done = s2.do_exercise(0)
    s += str(obs[1])


print s

print "Resetting both:"

s1.reset()
s2.reset()

s = ""
done = False
while not done:
    obs, reward, done = s1.do_exercise(0)
    s += str(obs[1])


print s


s = ""
done = False
while not done:
    obs, reward, done = s2.do_exercise(0)
    s += str(obs[1])


print s


print "Resetting, and doing them simultaneously:"


s1.reset()
s2.reset()

l1 = ""
l2 = ""
done = False
while not done:
    obs, reward, done = s1.do_exercise(0)
    l1 += str(obs[1])
    obs, reward, done = s2.do_exercise(0)
    l2 += str(obs[1])


print l1
print l2