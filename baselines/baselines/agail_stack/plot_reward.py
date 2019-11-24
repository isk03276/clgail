import matplotlib.pyplot as plt
import numpy as np
import sys

files = []

files.append(open("trpo_file_name1", 'r'))
files.append(open("trpo_file_name2", 'r'))
files.append(open("trpo_file_name3", 'r'))

files.append(open("gail_file_name4", 'r'))
files.append(open("gail_file_name5", 'r'))
files.append(open("gail_file_name6", 'r'))


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
      dr_list.append(dr_list[-1]*0.99+ reward*0.01)
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

trpo = [get_dr_list(_list[i], 1) for i in range(3)]
gail = [get_dr_list(_list[i], 1) for i in range(3, 6)]

plt.plot(meanPlot(trpo), label='TRPO(RL)')
plt.plot(meanPlot(gail), label='GAIL(IL)')
plt.xlabel('Learning Iteration')
plt.ylabel('Average of Success Rate')
plt.title('Stack')
plt.legend()
plt.show()

for f in files:
    f.close()

