from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.models import Model
import tensorflow as tf
import numpy as np


class SubGoalGenerator:
	def __init__(self, state_size):
		self.state_size = state_size
		self.discount_factor = 0.99
		self.lr = 0.003
	
		self.value_network = self.build_model()
		self.optimizer = self.optimizer()

	def build_model(self):
		input = Input(shape=(self.state_size,))
		fc = Dense(64, activation='relu')(input)
		fc = Dense(64, activation='relu')(fc)
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
		
	def train(self, total_states, total_rews, learning_iteration=500, batch_size=1):
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
					self.train_model(states.reshape(len(states), self.state_size), total_rews, False)
				else:
					self.train_model(states[:batch_size].reshape(batch_size, self.state_size), total_rews, False)
					self.train_model(states[batch_size:].reshape(len(states) - batch_size, self.state_size), total_rews, True)
			if iter % 100 == 0:
				print('training : ' + str(iter) + '/' + str(learning_iteration))

	def predict_state_score(self, state):
		"""
		input : list of state 
		output : min score of states
		"""
		value = self.value_network.predict(state.reshape(1, self.state_size))[0]
		return np.mean(value)

if __name__ == "__main__":
	demo = np.load('data/stack_rew.npz', allow_pickle=True)
	total_obs = demo['obs'][:50]
	total_acs = demo['acs'][:50]
	total_rews = demo['rets'][:50]
	subGoalGenerator = SubGoalGenerator(24)
	subGoalGenerator.train(total_obs,total_rews, learning_iteration = 10)
