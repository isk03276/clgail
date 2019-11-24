import matplotlib.pyplot as plt
import numpy as np
import sys

files = []

files.append(open("gail1", 'r'))
files.append(open("gail2", 'r'))
files.append(open("gail3", 'r'))
files.append(open("gail4", 'r'))
files.append(open("gail5", 'r'))
files.append(open("gail6", 'r'))
files.append(open("gail7", 'r'))
files.append(open("gail8", 'r'))
files.append(open("gail9", 'r'))
files.append(open("gail10", 'r'))

files.append(open("rl1", 'r'))
files.append(open("rl2", 'r'))
files.append(open("rl3", 'r'))
files.append(open("rl4", 'r'))
files.append(open("rl5", 'r'))
files.append(open("rl6", 'r'))
files.append(open("rl7", 'r'))
files.append(open("rl8", 'r'))
files.append(open("rl9", 'r'))
files.append(open("rl10", 'r'))

files.append(open("ex2_2_1", 'r'))
files.append(open("ex2_2_2", 'r'))
files.append(open("ex2_2_3", 'r'))
files.append(open("ex2_2_4", 'r'))
files.append(open("ex2_2_5", 'r'))
files.append(open("ex2_2_6", 'r'))
files.append(open("ex2_2_7", 'r'))
files.append(open("ex2_2_8", 'r'))
files.append(open("ex2_2_9", 'r'))
files.append(open("ex2_2_10", 'r'))

_list = [[] for _ in range(len(files))]


def read_reward(reward_list, f):
   while True:
      line = f.readline()
      if not line: break
      reward_list.append(float(line))

def get_dr_list(reward_list, mag):
   t = 0
   dr_list = [reward_list[0]]
   for reward in reward_list:
      if t > 6101:
          pass
      dr_list.append(dr_list[-1]*0.9+ reward*0.1)
      t += 1
   return dr_list

for l, f in zip(_list, files):
    read_reward(l, f)

def meanPlot(listlist):
    listlist = np.array(listlist)
    returnList = []
    smallestLen = len(listlist[0])
    for rlist in listlist:
        if len(rlist) < smallestLen:
            smallestLen = len(rlist)        

    for i in range(smallestLen):
        if i > 2000:
            break
        avg = 0
        for rewardlist in listlist:
            avg += rewardlist[i]
        avg /= len(listlist)
        returnList.append(avg)
    return returnList

gail = [get_dr_list(_list[i], 1) for i in range(10)]
rl = [get_dr_list(_list[i], 1) for i in range(10, 20)]
cl_gail = [get_dr_list(_list[i], 1) for i in range(20, 30)]

plt.plot(meanPlot(gail), label='GAIL')
plt.plot(meanPlot(rl), label='RL')
plt.plot(meanPlot(cl_gail), label='CL-GAIL')
plt.xlabel('Learning Iteration')
plt.ylabel('Average of Success Rate')
plt.title('Stack')
plt.legend()
plt.show()

for f in files:
    f.close()

