from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
import tensorflow as tf
import numpy as np


class SubGoalGenerator:
	def __init__(self, state_size):
		self.state_size = state_size
		self.discount_factor = 0.99
		self.no_op_steps = 30
		self.lr = 0.005
	
		self.value_network = self.build_model()
		self.optimizer = self.optimizer()

	def build_model(self):
		value_network = Sequential()
		value_network.add(Dense(64, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
		value_network.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
		value_network.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))

		value_network.summary()

		return value_network

	def optimizer(self):
		target = K.placeholder(shape=(None, ))
	
		loss = K.mean(K.square(target - self.value_network.output))

		optimizer = Adam(lr = self.lr)
		updates = optimizer.get_updates(self.value_network.trainable_weights, [], loss)
		train = K.function([self.value_network.input, target], [], updates=updates)

		return train


	def train_model(self, state, reward, next_state, done):
		value = self.value_network.predict(state)[0]

		if done:
			target = [reward]
		else:
			next_value = self.value_network.predict(next_state)[0]
			target = reward + self.discount_factor * next_value

		self.optimizer([state, target])

	def train(self, states, learning_iteration=5000, batch_size=1):
		"""
		input : expert demo data trajectory
		to-do : batch learning
		"""
		#expert_data = np.load('stack.npz', allow_pickle=True)
		#states = expert_data['obs'][:100]
		self.state_size = states[0][0].shape[0]
		cur_idx = 0
		batch_size = 1
		
		dones = []
		rews = []
		next_states = []
		for ep in range(len(states)):
			next_state = list(states[ep][1:])
			next_state.append(next_state[-1])
			for step in range(len(states[ep])):
				dones.append(False)
				rews.append(0)
			dones[-1] = True
			rews[-1] = 1
				
			next_states.append(np.array(next_state))

		dones = np.array(dones)
		rews = np.array(rews)
		next_states = np.array(next_states)

		states = np.vstack(states)
		next_states = np.vstack(next_states)

		assert len(states) == len(dones) == len(rews) == len(next_states)	

		traj_size = len(states)
		
		#train
		for iter in range(learning_iteration):
			cur_idx = (cur_idx + batch_size) % traj_size
			
			self.train_model(states[cur_idx: cur_idx + batch_size].reshape(batch_size, state_size), 
			rews[cur_idx: cur_idx + batch_size], next_states[cur_idx: cur_idx + batch_size].reshape(batch_size, state_size), dones[cur_idx: cur_idx + batch_size])
			
			if iter % 100 == 0:
				print('training : ' + str(iter) + '/' + str(learning_iteration))

	def create_sub_goals(self, obs, num_of_goals):
		"""
		input : an expert demo data trajectory
		"""
		state_value_list = []
		sub_goal_list = []
		for ob in obs:
			state_value_list.append(self.value_network.predict(ob.reshape(1, self.state_size))[0])
		state_value_list[0] += state_value_list[1] #don't use initial state
		state_value_list[-1] += state_value_list[-2] #don't use final state
		for i in range(num_of_goals):
			start_idx = i*int(len(obs)/num_of_goals)
			end_idx = start_idx + int(len(obs)/num_of_goals)
			if end_idx > len(obs):
				sub_goal_list.append(np.argmin(np.array(state_value_list[start_idx:])))
			else:
				sub_goal_list.append(np.argmin(np.array(state_value_list[start_idx:end_idx])))
		return sub_goal_list

if __name__ == "__main__":
	subGoalGenerator = SubGoalGenerator(24)