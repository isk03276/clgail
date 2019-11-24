from keras.layers import Dense, Input, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.models import Model
import tensorflow as tf
import numpy as np


class SubGoalGenerator:
	def __init__(self, state_size):
		print("what the################")
		self.state_size = state_size
		self.discount_factor = 0.99
		self.lr = 0.003
	
		self.value_network = self.build_model()
		self.optimizer = self.optimizer()

	def build_model(self):
		input = Input(shape=(self.state_size,))
		fc = Dense(64, activation='relu')(input)
		fc = Dense(64, activation='relu')(fc)
                fc.add(Dropout(0.3))
		value = Dense(1, activation='linear')(fc)

		value_network = Model(inputs=input, outputs=value)
		value_network._make_predict_function()

		value_network.summary()

		return value_network

	def optimizer(self):
		
		target = K.placeholder(shape=(None, ))
	
		loss = K.mean(K.square(target - self.value_network.output))

		optimizer = Adam(lr = self.lr)
		updates = optimizer.get_updates(self.value_network.trainable_weights, [], loss)
		train = K.function([self.value_network.input, target], [], updates=updates)

		return train


	def train_model(self, states, rewards, done):
		targets = np.zeros_like(rewards)
		running_add = 0

		if not done:
			running_add = self.value_network.predict(states[-1].reshape(1, self.state_size))[0]

		for t in reversed(range(0, len(rewards))):
			running_add = running_add * self.discount_factor + rewards[t]
			targets[t] = running_add
		
		self.optimizer([states, targets])
		
	def train(self, total_states, learning_iteration=100, batch_size=1):
		"""
		input : expert demo data trajectory
		to-do : batch learning
		"""
		#expert_data = np.load('stack.npz', allow_pickle=True)
		#states = expert_data['obs'][:100]
		self.state_size = total_states[0][0].shape[0]
		cur_idx = 0
		batch_size = 15

		for iter in range(learning_iteration):
			for states in total_states:
				if len(states) <= batch_size:
					rews = np.zeros_like(states)
					try:
						rews[-1] = 1
					except: 
						pass#print(rews)
						#print(rews.shape)
					self.train_model(states.reshape(len(states), self.state_size), rews, False)
				else:
					rews = np.zeros(batch_size)
					self.train_model(states[:batch_size].reshape(batch_size, self.state_size), rews, False)
					rews = np.zeros(len(states) - batch_size)
					rews[-1] = 1
					self.train_model(states[batch_size:].reshape(len(states) - batch_size, self.state_size), rews, True)
			if iter % 100 == 0:
				print('training : ' + str(iter) + '/' + str(learning_iteration))

	def create_sub_goals(self, total_states, num_of_goals):
		"""
		input : total expert demo data trajectory
		output : sub goals
		"""
		return_list = []
		#print(total_states.shape)
		for states in total_states:
			state_value_list = []
			for state in states:
				state_value_list.append(self.value_network.predict(state.reshape(1, self.state_size))[0][0])
			state_value_list[0] -= abs(state_value_list[1]) #don't use initial state
			state_value_list[-1] -= abs(state_value_list[-2]) #don't use final state
			sub_goal_list = np.sort(np.argsort(state_value_list)[-3:])
			"""
			for i in range(num_of_goals):
				sub_goal_list
				start_idx = i*int(len(states)/num_of_goals)
				end_idx = start_idx + int(len(states)/num_of_goals)
				if end_idx > len(states):
					sub_goal_list.append(start_idx + np.argmax(np.array(state_value_list[start_idx:])))
				else:
					sub_goal_list.append(start_idx + np.argmax(np.array(state_value_list[start_idx:end_idx])))
			"""
			return_list.append(sub_goal_list)
		return return_list
	def predict_state_score(self, state):
		"""
		input : list of state 
		output : min score of states
		"""
		value = self.value_network.predict(state.reshape(1, self.state_size))[0]
		return np.mean(value)

if __name__ == "__main__":
	subGoalGenerator = SubGoalGenerator(24)
