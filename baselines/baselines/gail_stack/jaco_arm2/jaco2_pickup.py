import os
import mujoco_py
import numpy as np
from gym.utils import seeding
import time
from random import *
import math
from gym import spaces

class Jaco2PickUpEnv():
    def __init__(self,
                 width=0,
                 height=0,
                 frame_skip=1,
                 rewarding_distance=1.0):
        self.frame_skip = frame_skip
        self.width = width
        self.height = height

        #self.observation_space = 21
        self.observation_space = spaces.Box(low=-50, high=50, shape=(21,))
        #self.action_space = 9
        self.action_space = spaces.Box(low=-50, high=50, shape=(9,))
        self.reward_range = 10
        self.metadata = []

        # Instantiate Mujoco model
        model_path = "jaco_arm2_pickup.xml"
        fullpath = os.path.join(
            os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model)

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

        #self.reset_target()
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

        self.setting_pose = None

        self.initial_states_dist = np.load('data/jaco2_initial_states8.npz', allow_pickle=True)['initial_states']
        self.initial_states_list = self.initial_states_dist[:, 0]
        
        

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

    def set_target_pos(self, pos):
        assert len(pos) == 3
        self.setting_pose = pos
        
    def reset(self):
        self.episode_now = 0
        initial_state = choice(self.initial_states_list)
        self.reset_backup()
        self.sim.set_state_from_flattened(initial_state)
        self.sim.forward()

        return self.get_obs()


    def reset_backup(self):
        # Random initial position of Jaco
        # qpos = self.init_qpos + np.random.randn(self.sim.nv)

        #  Fixed initial position of Jaco
        self.episode_now = 0
        self.init_qpos[0] = -3.14
        self.init_qpos[1] = 3.14
        self.init_qpos[2] = 1.7
        self.init_qpos[3] = 0.
        self.init_qpos[4] = 0.
        self.init_qpos[5] = 0.
        self.init_qpos[6] = 0.
        self.init_qpos[7] = 0.
        self.init_qpos[8] = 0.
        self.init_qvel[0] = 0.
        self.init_qvel[1] = 0.
        self.init_qvel[2] = 0.
        self.init_qvel[3] = 0.
        self.init_qvel[4] = 0.
        self.init_qvel[5] = 0.
        self.init_qvel[6] = 0.
        self.init_qvel[7] = 0.
        self.init_qvel[8] = 0.

        if type(self.setting_pose) == type(None):
            self.init_qpos[9:12] = self.get_target_randompos()
        else:
            self.init_qpos[9:12] = self.setting_pose
            self.setting_pose = None
            
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
        self.finger_cener = (self.sim.data.get_body_xpos("j2n6s300_link_finger_1") + self.sim.data.get_body_xpos("j2n6s300_link_finger_2") + self.sim.data.get_body_xpos("j2n6s300_link_finger_3")) / 3

        self.first_imitation_flag = False
        self.first_imitation_count = 0

        return self.get_obs()

    def get_target_randompos(self):

        while(True):
            pos_first = random()
            if pos_first > 0.48 and pos_first < 0.498:
                break
        while(True):
            pos_second = random() - 0.5
            if pos_second > -0.197 and pos_second < 0.1398:
                break
        pos_third = 0.035

        return [pos_first, pos_second, pos_third]
        #self.sim.model.geom_pos[0] = [pos_first, pos_second, pos_third]

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
        return np.concatenate((self._get_obs_joint(), self.sim.data.get_body_xpos("target1"))) #pickup
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
        for mat in self.sim.data.get_body_xmat("j2n6s300_link_finger_1"):
            self.first_finger_vector.append(mat[0])

        self.second_finger_vector = []
        for mat in self.sim.data.get_body_xmat("j2n6s300_link_finger_2"):
            self.second_finger_vector.append(mat[0])

        self.third_finger_vector = []
        for mat in self.sim.data.get_body_xmat("j2n6s300_link_finger_3"):
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

        if self.episode_now > 100:
            done = True

        # if continuous reward
        # reward = float((np.mean(dist)**-1)*0.1)
        reward = 0

        #################################################
        ## first phase ##
        #####################

        center = (self.sim.data.get_body_xpos("j2n6s300_link_finger_1") + self.sim.data.get_body_xpos("j2n6s300_link_finger_2") + self.sim.data.get_body_xpos("j2n6s300_link_finger_3")) / 3
        centerd = np.linalg.norm(center - self.sim.data.get_body_xpos("target1"))


        # go lift target1
        # So hard
        if centerd < 0.1:
            #reward += 0.1
            #reward += abs(self.sim.data.get_body_xpos("target1")[2] - 0.035)
            #reward += sum(self.sim.data.qpos.flat[6:9])
            if self.sim.data.get_body_xpos("target1")[2] > 0.04:
                pass#reward += abs(self.sim.data.get_body_xpos("target1")[2] - 0.025) #pass#reward += self.sim.data.get_body_xpos("target1")[2] # + self.sim.data.get_body_xpos("target1")[2]
                #reward += (self.sim.data.get_body_xpos("target1")[2] - 0.03) * 3000
        
            if self.sim.data.get_body_xpos("target1")[2] > 0.25:
                reward += 1#1000#0.6
                self.second_phase = True
                self.success = True
                done = True

        

        self.do_simulation(a)
        self.sum_reward += reward
        self.episode_now += 1

        #print(reward)

        return self.get_obs(), reward, done, self.success
