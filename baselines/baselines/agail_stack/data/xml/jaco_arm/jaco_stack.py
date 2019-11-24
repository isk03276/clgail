import os
import mujoco_py
import numpy as np
from gym.utils import seeding
import time
from random import *
import math
from gym import spaces

class JacoStackEnv():
    def __init__(self,
                 width=0,
                 height=0,
                 frame_skip=1,
                 rewarding_distance=1.0,
                 sim = None):
        self.frame_skip = frame_skip
        self.width = width
        self.height = height

        #self.observation_space = 21
        self.observation_space = spaces.Box(low=-50, high=50, shape=(24,))
        #self.action_space = 9
        self.action_space = spaces.Box(low=-50, high=50, shape=(9,))
        self.reward_range = 10
        self.metadata = []

        # Instantiate Mujoco model
        if sim == None:
            model_path = "jaco_stack.xml"
            fullpath = os.path.join(
                os.path.dirname(__file__), "assets", model_path)
            if not os.path.exists(fullpath):
                raise IOError("File %s does not exist" % fullpath)
            model = mujoco_py.load_model_from_path(fullpath)
            self.sim = mujoco_py.MjSim(model)
        else:
            self.sim = sim

        self.init_state = self.sim.get_state()
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        # Setup actuators
        self.actuator_bounds = self.sim.model.actuator_ctrlrange
        self.actuator_low = self.actuator_bounds[:, 0]
        self.actuator_high = self.actuator_bounds[:, 1]
        self.actuator_ctrlrange = self.actuator_high - self.actuator_low
        self.num_actuators = len(self.actuator_low)

        # init model_data_ctrl
        self.null_action = np.zeros(self.num_actuators)
        self.sim.data.ctrl[:] = self.null_action

        self.seed()

        self.sum_reward = 0
        self.rewarding_distance = rewarding_distance
        self.success = False

        # Target position bounds
        self.target_bounds = np.array(((0.3, 0.4), (-0.5, 0.5), (0.05, 0.05)))
        self.target_reset_distance = 0.2

        #self.reset_target()
        self.goal_list = [[0.6, -0.3, 0.3], [0.6, 0.3, 0.3], [0.6, 0., 0.3]]
        #self.goal_list = [[0.6, -0.4, 0.5]]
        self.episode = 0
        self.goal = [0.6, 0.3, 0.3]

        self.first_imitation_flag = False
        self.first_imitation_count = 0

        self.second_imitation_flag = False
        self.second_imitation_count = 0

        self.second_phase = False
        self.third_phase = False

        #_file = np.load("data/stack_data.npz")
        #self.file_action = _file["acs"]
        self.file_index = 0
        self.sr = 0
        
        

        # Setup discrete action space
        #self.control_values = self.actuator_ctrlrange * control_magnitude

        #self.num_actions = 5
        # self.action_space = [list(range(self.num_actions))
        #                      ] * self.num_actuators
        # self.observation_space = ((0, ), (height, width, 3),
        #                           (height, width, 3))

        #self.reset_target()
        self.reset()


    def setEpisode(self, episode):
        self.episode = episode

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_qpos_qvel(self, qpos, qvel):
        assert qpos.shape == (self.sim.model.nq, ) and qvel.shape == (
            self.sim.model.nv, )
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()

    def reset(self):
        # Random initial position of Jaco
        # qpos = self.init_qpos + np.random.randn(self.sim.nv)

        #  Fixed initial position of Jaco
        self.episode_now = 0
        qpos = self.init_qpos
        qvel = self.init_qvel

        #self.rewarding_distance = max(0.1, self.rewarding_distance * (0.999 ** self.episode))
        #self.rewarding_distance = max(0.07, 0.4 * (0.999 ** (self.episode/2)))
        #print(self.rewarding_distance)
        #print(self.rewarding_distance)

        # random object position start of episode
        #self.reset_target()

        # set initial joint positions and velocities
        self.success = False
        self.set_qpos_qvel(qpos, qvel)
        self.init_finger_g = (self.sim.data.get_body_xpos("jaco_link_finger_1") + self.sim.data.get_body_xpos("jaco_link_finger_2") + self.sim.data.get_body_xpos("jaco_link_finger_3")) / 3
        self.old_finger_g = (self.sim.data.get_body_xpos("jaco_link_finger_1") + self.sim.data.get_body_xpos("jaco_link_finger_2") + self.sim.data.get_body_xpos("jaco_link_finger_3")) / 3

        self.first_imitation_flag = False
        self.first_imitation_count = 0

        return self.get_obs()

    def reset_target(self):

        while(True):
            pos_first = random()
            if pos_first > 0.3 and pos_first < 0.5:
                break
        while(True):
            pos_second = random() - 0.5
            if pos_second > -0.3 and pos_second < 0.5:
                break
        pos_third = 0.031

        self.goal = [pos_first, pos_second, pos_third]
        self.sim.model.geom_pos[1] = [pos_first, pos_second, pos_third]

        # self.goal_index = random.randrange(0, 3)
        # self.goal = self.goal_list[self.goal_index]
        # self.sim.model.geom_pos[1] = self.goal[:]

    def zero_shot(self):
        self.goal = [0.6, 0.0, 0.6]
        self.sim.model.geom_pos[1] = [0.6, 0.0, 0.6]

    def render(self, camera_name=None):
        rgb = self.sim.render(
            width=self.width, height=self.height, camera_name=camera_name)
        return rgb

    def _get_obs_joint(self):
        return np.concatenate(
            [self.sim.data.qpos.flat[:9], self.sim.data.qvel.flat[:9]])

    def _get_obs_rgb_view1(self):
        obs_rgb_view1 = self.render(camera_name='view1')
        return obs_rgb_view1

    def _get_obs_rgb_view2(self):
        obs_rgb_view2 = self.render(camera_name='view2')
        return obs_rgb_view2

    def get_obs(self):
        # return (self._get_obs_joint(), self._get_obs_rgb_view1(),
        #         self._get_obs_rgb_view2())
        #return self._get_obs_joint()
        return np.concatenate((self._get_obs_joint(), self.sim.data.get_body_xpos("target1"), self.sim.data.get_body_xpos("target2")))
        #return self._get_obs_joint()
        

    def do_simulation(self, ctrl):
        '''Do one step of simulation, taking new control as target

        Arguments:
            ctrl {np.array(num_actuator)}  -- new control to send to actuators
        '''
        ctrl = np.min((ctrl, self.actuator_high), axis=0)
        ctrl = np.max((ctrl, self.actuator_low), axis=0)

        self.sim.data.ctrl[:] = ctrl

        for _ in range(self.frame_skip):
            self.sim.step()

    def get_finger_vector(self):

        self.first_finger_vector = []
        for mat in self.sim.data.get_body_xmat("jaco_link_finger_1"):
            self.first_finger_vector.append(mat[0])

        self.second_finger_vector = []
        for mat in self.sim.data.get_body_xmat("jaco_link_finger_2"):
            self.second_finger_vector.append(mat[0])

        self.third_finger_vector = []
        for mat in self.sim.data.get_body_xmat("jaco_link_finger_3"):
            self.third_finger_vector.append(mat[0])


    def calc_cos(self, vector1, vector2):

        vector1_size = 0
        for vector in vector1:
            vector1_size += vector*vector
        vector1_size = math.sqrt(vector1_size)

        vector2_size = 0
        for vector in vector2:
            vector2_size += vector*vector
        vector2_size = math.sqrt(vector2_size)

        dotproduct = 0
        for i in range(3):
            dotproduct += vector1[i] * vector2[i]

        cos = dotproduct / (vector1_size * vector2_size)

        return cos


    def next_action(self):
        pass

    def calc_angles_reward(self):

        cos1 = self.calc_cos(self.fingercenter_to_object_vector, self.first_finger_vector)
        cos2 = self.calc_cos(self.fingercenter_to_object_vector, self.second_finger_vector)
        cos3 = self.calc_cos(self.fingercenter_to_object_vector, self.third_finger_vector)

        return cos1 + cos2 + cos3


    # @profile(immediate=True)
    def step(self, a):
        dist = np.zeros(3)
        done = False
        new_control = np.copy(a).flatten()

        if self.episode_now > 64:
            done = True

        # if continuous reward
        # reward = float((np.mean(dist)**-1)*0.1)
        reward = 0

        #################################################
        ## first phase ##
        #####################

        center = (self.sim.data.get_body_xpos("jaco_link_finger_1") + self.sim.data.get_body_xpos("jaco_link_finger_2") + self.sim.data.get_body_xpos("jaco_link_finger_3")) / 3
        centerd = np.linalg.norm(center - self.sim.data.get_body_xpos("target1"))

        if center[0] < 0.3:
            pass#done = True

        self.fingercenter_to_object_vector = self.sim.data.get_body_xpos("target1") - center
        self.get_finger_vector()


        finger1_tip_position = [i/10 for i in self.first_finger_vector] + self.sim.data.get_body_xpos("jaco_link_finger_1")
        finger2_tip_position = [i/10 for i in self.second_finger_vector] + self.sim.data.get_body_xpos("jaco_link_finger_2")
        finger3_tip_position = [i/10 for i in self.third_finger_vector] + self.sim.data.get_body_xpos("jaco_link_finger_3")
        finger_center = (finger1_tip_position + finger2_tip_position + finger3_tip_position + center) / 4
        hand_center = (center*2 + finger_center) / 3

        #self.sim.model.geom_pos[1] = self.sim.data.get_body_xpos("target2") + [0, 0, 0.07]

        hand_d = np.linalg.norm(hand_center - self.sim.data.get_body_xpos("target1"))

        if self.first_imitation_count == 4:
            self.first_imitation_flag = False
            self.first_imitation_count = 100 # be big and never again

        if self.first_imitation_flag and self.first_imitation_count < 15:
            self.first_imitation_count += 1
        elif hand_d < 0.07 and self.first_imitation_count < 15:
            self.first_imitation_flag = True
        


        
        # if self.second_imitation_count == 5:
        #     self.second_imitation_flag = False
        #     self.second_imitation_count = 100 # be big and never again

        # if self.second_imitation_flag and self.second_imitation_count < 15:
        #     self.second_imitation_count += 1
        # elif distance_target1_ontarget2 < 0.08 and self.second_imitation_count < 15:
        #     self.second_imitation_flag = True


        #self.sim.model.geom_pos[2] = finger3_tip_position


        # for d in dist:
        #     reward -= d

        dist_sum = 0


        # easy
        # reward -= hand_d
        if hand_d > 0.30:
            pass#done = True

        #reward += (1/(hand_d+0.1))
        #     #reward += 30
        #     #done = True

        #print(self.calc_angles_reward())
        finger1_tip_to_target_distance = np.linalg.norm(finger1_tip_position - self.sim.data.get_body_xpos("target1"))
        finger2_tip_to_target_distance = np.linalg.norm(finger2_tip_position - self.sim.data.get_body_xpos("target1"))
        finger3_tip_to_target_distance = np.linalg.norm(finger3_tip_position - self.sim.data.get_body_xpos("target1"))
        fintertips_to_target_distancesum = finger1_tip_to_target_distance + finger2_tip_to_target_distance + finger3_tip_to_target_distance

        # if dist[0] < 0.12 and dist[1] < 0.12 and dist[2] < 0.12:
        #     reward += 10 / (fintertips_to_target_distancesum)


        # go reach target1
        # close and 
        # reward += 0.1 / (fintertips_to_target_distancesum)

        # go lift target1
        # So hard
        if hand_d < 0.10:
            #reward += 1
            if self.sim.data.get_body_xpos("target1")[2] > 0.06 and self.sim.data.get_body_xpos("target1")[2] < 0.2:
                pass#reward += 0.5 #get demo
                #reward += (self.sim.data.get_body_xpos("target1")[2] - 0.03) * 3000
        if self.sim.data.get_body_xpos("target1")[2] > 0.11:
            #reward += 1#0.6
            self.second_phase = True

        # if self.sim.data.get_body_xpos("target3")[2] > 0.3:
        #     self.third_phase = True
                

        # 3phase
        # should be closer 0.5 0.8 0.5pos("target1")[2] > 0.06 and self.sim.data.get_body_xpos("target1")[2] < 0.2:
        # if np.linalg.norm([0.5, 0.6, 0.43] - self.sim.                reward += (self.sim.data.get_body_xpos("target1")[2] - 0.03) * 3000data.get_body_xpos("target")) < 0.20:
        #     if self.sim.data.get_body_xpos("target")[2] > 0.43:
        #         if finger1_tip_to_target_distance < 0.10 and finger2_tip_to_target_distance < 0.10 and finger3_tip_to_target_distance < 0.10:
        #             reward += 10000 * (1/(np.linalg.norm([0.5, 0.6, 0.43] - self.sim.data.get_body_xpos("target")) + 0.01))

        # collision
        #print(self.sim.data.active_contacts_efc_pos)
        # for collision in self.sim.data.active_contacts_efc_pos:
        #     if abs(collision) > 0.001:
        #         # collision!
        #         reward -= 50
        #         done = True
        #         print(collision)
        # if len(self.sim.data.active_contacts_efc_pos) > 7 and len(self.sim.data.active_contacts_efc_pos) < 17:
        #     done = True



        # if centerd == self.prev_centerd:
        #     reward += 10

        #
        # reward += (self.sim.data.get_body_xpos("target")[2] - 0.030989) * 100
        # if self.sim.data.get_body_xpos("target")[2] - 0.030989 > 0.05:return
        #     reward += (self.sim.data.get_body_xpos("target")[2] - 0.030989) * 1000

        # self.prev_centerd = centerd

        #  Fixed initial position of Jaco
        # if np.linalg.norm([0.5, 0.6, 0.43] - self.sim.data.get_body_xpos("target")) < 0.1:
        #     success = True


        #################################################
        ## second phase ##
        #####################

        # z ~ 0.105
        # 3
        distance_target1_ontarget2 = np.linalg.norm(self.sim.data.get_body_xpos("target1") - (self.sim.data.get_body_xpos("target2") + [0, 0, 0.07]))
        if distance_target1_ontarget2 < 0.05 and self.second_phase:
            if (0.100 < self.sim.data.get_body_xpos("target1")[2] < 0.110):
                reward += 1#2
                done = True
                self.success = True

        
        # # 3'
        # fly_high = np.linalg.norm(self.sim.data.get_body_xpos("target1") - (self.sim.data.get_body_xpos("target2") + [0, 0, 0.30]))
        # if distance_target1_ontarget2 < 0.05:
        #     if (0.100 < self.sim.data.get_body_xpos("target1")[2] < 0.110):
        #         reward += (100000 / fly_high+0.1)


        # # 4
        # distance_target3_hand = np.linalg.norm(hand_center - self.sim.data.get_body_xpos("target3"))
        # if (0.100 < self.sim.data.get_body_xpos("target1")[2] < 0.110):
        #     if distance_target1_ontarget2 < 0.05:
        #         reward += (1000000/(distance_target3_hand+0.1))

        # # 5 lift
        # if distance_target3_hand < 0.10:
        #     if distance_target1_ontarget2 < 0.05:
        #         if self.sim.data.get_body_xpos("target3")[2] > 0.06 and self.sim.data.get_body_xpos("target3")[2] < 0.3:
        #             reward += (self.sim.data.get_body_xpos("target3")[2] - 0.03) * 10000000

        # # 6
        # distance_target3_ontarget2 = np.linalg.norm(self.sim.data.get_body_xpos("target3") - (self.sim.data.get_body_xpos("target2") + [0, 0, 0.14]))
        # if distance_target3_ontarget2 < 0.3 and self.third_phase:
        #     if distance_target1_ontarget2 < 0.05:
        #         if (0.100 < self.sim.data.get_body_xpos("target1")[2] < 0.110):
        #             if (0.170 < self.sim.data.get_body_xpos("target3")[2] < 0.180):
        #                 reward += (100000000 / distance_target3_ontarget2+0.1)
        
        # ele = self.file_action[self.file_index]
        # self.file_index += 1
        # a = ele

        self.do_simulation(a)
        self.sum_reward += reward
        self.episode_now += 1

        #print(reward)

        return self.get_obs(), reward, done, self.success
