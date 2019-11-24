import matplotlib.pyplot as plt
import numpy as np
import sys

files = []

files.append(open("ex2_1_1", 'r'))
files.append(open("ex2_1_2", 'r'))
files.append(open("ex2_1_3", 'r'))
files.append(open("ex2_1_4", 'r'))
files.append(open("ex2_1_5", 'r'))

files.append(open("ex2_2_1", 'r'))
files.append(open("ex2_2_2", 'r'))
files.append(open("ex2_2_3", 'r'))
files.append(open("ex2_2_4", 'r'))
files.append(open("ex2_2_5", 'r'))

files.append(open("ex2_3_1", 'r'))
files.append(open("ex2_3_2", 'r'))
files.append(open("ex2_3_3", 'r'))
files.append(open("ex2_3_4", 'r'))
files.append(open("ex2_3_5", 'r'))

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
        if i > 10000:
            break
        avg = 0
        for rewardlist in listlist:
            avg += rewardlist[i]
        avg /= len(listlist)
        returnList.append(avg)
    return returnList

sr1 = [get_dr_list(_list[i], 1) for i in range(5)]
sr2 = [get_dr_list(_list[i], 1) for i in range(5, 10)]
policy = [get_dr_list(_list[i], 1) for i in range(10, 15)]

plt.plot(meanPlot(sr1), label='1')
plt.plot(meanPlot(sr2), label='2')
plt.plot(meanPlot(policy), label='3')
plt.xlabel('Learning Iteration')
plt.ylabel('Average of Success Rate')
plt.title('Stack')
plt.legend()
plt.show()

for f in files:
    f.close()

