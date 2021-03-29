import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# BCQ's code from https://github.com/sfujim/BCQ

# Used for Atari
class Conv_Q(nn.Module):
	def __init__(self, frames, num_actions, device):
		super(Conv_Q, self).__init__()
		self.device = device
		self.num_actions = num_actions
		self.c1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
		self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

		# Q-learning
		self.q1 = nn.Linear(3136, 512)
		# head 1
		self.q2_1 = nn.Linear(512, num_actions)
		# head 2
		self.q2_2 = nn.Linear(512, num_actions)
		# head 3
		self.q2_3 = nn.Linear(512, num_actions)
		# head 4
		self.q2_4 = nn.Linear(512, num_actions)
		# head 5
		self.q2_5 = nn.Linear(512, num_actions)

		# imitation (BC)
		self.i1 = nn.Linear(3136, 512)
		# head 1
		self.i2_1 = nn.Linear(512, num_actions)
		# head 2
		self.i2_2 = nn.Linear(512, num_actions)
		# head 3
		self.i2_3 = nn.Linear(512, num_actions)
		# head 4
		self.i2_4 = nn.Linear(512, num_actions)
		# head 5
		self.i2_5 = nn.Linear(512, num_actions)


	def compute_Q(self, state, need_std=False):
		c = F.relu(self.c1(state))
		c = F.relu(self.c2(c))
		c = F.relu(self.c3(c))

		q = F.relu(self.q1(c.reshape(-1, 3136)))
		i = F.relu(self.i1(c.reshape(-1, 3136)))
		# logits
		i_1, i_2, i_3, i_4, i_5 = self.i2_1(i), self.i2_2(i), self.i2_3(i), self.i2_4(i), self.i2_5(i)
		i = (i_1 + i_2 + i_3 + i_4 + i_5) / 5.0

		# compute ensemble Q-values (batch, action dim)
		q2_1, q2_2, q2_3, q2_4, q2_5 = self.q2_1(q), self.q2_2(q), self.q2_3(q), self.q2_4(q), self.q2_5(q)
		q2 = (q2_1 + q2_2 + q2_3 + q2_4 + q2_5) / 5.0
		
		# compute action std (batch, action dim)
		# 1. probability std
		# std_matrix = self._compute_action_std(F.softmax(i_1, dim=1), F.softmax(i_2, dim=1), 
		# 	F.softmax(i_3, dim=1), F.softmax(i_4, dim=1), F.softmax(i_5, dim=1)) if need_std==True else None
		# 2. logits std
		std_matrix = self._compute_action_std(i_1, i_2, i_3, i_4, i_5) if need_std==True else None
		return q2, F.log_softmax(i, dim=1), i, std_matrix


	def _compute_action_std(self, i_1, i_2, i_3, i_4, i_5):
		logits_matrix = torch.stack([i_1, i_2, i_3, i_4, i_5])
		std_matrix = [] # (batch, action dim)

		for i_sample in range(logits_matrix.shape[1]):
			action_std = []
			for j_action in range(logits_matrix.shape[2]):
				action_std.append(logits_matrix[:, i_sample, j_action].std())	# use std
				# action_std.append(logits_matrix[:, i_sample, j_action].var())	# use var
			std_matrix.append(action_std)
		return torch.Tensor(std_matrix).to(self.device)		# (batch, action dim)


	def forward(self, state, need_std=False):
		'''
		return: Q-values, log(probabilites), logits
		'''
		c = F.relu(self.c1(state))
		c = F.relu(self.c2(c))
		c = F.relu(self.c3(c))

		q = F.relu(self.q1(c.reshape(-1, 3136)))
		batch = q.shape[0]
		i = F.relu(self.i1(c.reshape(-1, 3136)))

		# compute ensemble logits (batch, action dim)
		i_1, i_2, i_3, i_4, i_5 = self.i2_1(i), self.i2_2(i), self.i2_3(i), self.i2_4(i), self.i2_5(i)
		# compute ensemble Q-values (batch, action dim)
		q2_1, q2_2, q2_3, q2_4, q2_5 = self.q2_1(q), self.q2_2(q), self.q2_3(q), self.q2_4(q), self.q2_5(q)
		
		# random ensemble mixture
		alpha = torch.Tensor(batch, 5).uniform_(0, 1).to(self.device)
		d = alpha.sum(dim=1)
		alpha = torch.stack([x/d[idx] for idx, x in enumerate(alpha)]).to(self.device)
		# alpha (batch, 5), alpha 1 + alpha1 2 + ... + alpha 5 = 1.0

		# random ensemble logits
		i = alpha[:, 0].view(-1, 1)*i_1 + alpha[:, 1].view(-1, 1)*i_2 + alpha[:, 2].view(-1, 1)*i_3 + alpha[:, 3].view(-1, 1)*i_4 + alpha[:, 4].view(-1, 1)*i_5
		# random ensemble q
		q2 = alpha[:, 0].view(-1, 1)*q2_1 + alpha[:, 1].view(-1, 1)*q2_2 + alpha[:, 2].view(-1, 1)*q2_3 + alpha[:, 3].view(-1, 1)*q2_4 + alpha[:, 4].view(-1, 1)*q2_5

		# compute action std (batch, action dim)
		# # 1. probability std
		# std_matrix = self._compute_action_std(F.softmax(i_1, dim=1), F.softmax(i_2, dim=1), 
		# 	F.softmax(i_3, dim=1), F.softmax(i_4, dim=1), F.softmax(i_5, dim=1))
		# 2. logits std
		std_matrix = self._compute_action_std(i_1, i_2, i_3, i_4, i_5) if need_std==True else None
		return q2, F.log_softmax(i, dim=1), i, std_matrix


# Used for Box2D / Toy problems
class FC_Q(nn.Module):
	def __init__(self, state_dim, num_actions):
		super(FC_Q, self).__init__()
		self.q1 = nn.Linear(state_dim, 256)
		self.q2 = nn.Linear(256, 256)
		self.q3 = nn.Linear(256, num_actions)

		self.i1 = nn.Linear(state_dim, 256)
		self.i2 = nn.Linear(256, 256)
		self.i3 = nn.Linear(256, num_actions)		


	def forward(self, state):
		q = F.relu(self.q1(state))
		q = F.relu(self.q2(q))

		i = F.relu(self.i1(state))
		i = F.relu(self.i2(i))
		i = F.relu(self.i3(i))
		return self.q3(q), F.log_softmax(i, dim=1), i


class discrete_BCQ(object):
	def __init__(
		self, 
		is_atari,
		num_actions,
		state_dim,
		device,
		BCQ_threshold=0.3,
		discount=0.99,
		optimizer="Adam",
		optimizer_parameters={},
		polyak_target_update=False,
		target_update_frequency=8e3,
		tau=0.005,
		initial_eps = 1,
		end_eps = 0.001,
		eps_decay_period = 25e4,
		eval_eps=0.001,
	):
	
		self.device = device

		# Determine network type
		self.Q = Conv_Q(state_dim[0], num_actions, device).to(self.device) if is_atari else FC_Q(state_dim, num_actions).to(self.device)
		self.Q_target = copy.deepcopy(self.Q)
		self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

		self.discount = discount

		# Target update rule
		self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
		self.target_update_frequency = target_update_frequency
		self.tau = tau

		# Decay for eps
		self.initial_eps = initial_eps
		self.end_eps = end_eps
		self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

		# Evaluation hyper-parameters
		self.state_shape = (-1,) + state_dim if is_atari else (-1, state_dim)
		self.eval_eps = eval_eps
		self.num_actions = num_actions

		# Threshold for "unlikely" actions
		self.threshold = BCQ_threshold

		# Number of training iterations
		self.iterations = 0


	def select_action(self, state, eval=False):
		# Select action according to policy with probability (1-eps)
		# otherwise, select random action
		if np.random.uniform(0,1) > self.eval_eps:
			with torch.no_grad():
				state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
				# how to use std (without gradient) TODO
				q, imt, i, std_matrix = self.Q.compute_Q(state, need_std=False)	# use average Q-values during the test
				imt = imt.exp()
				imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()
				# imt is a matrix (batch, action dim) with 0 or 1 values
				# Use large negative number to mask actions from argmax
				# at the index that meets the condition(e.g., > threshold) -> taking Q-value
				# otherwise -> taking large negative number (e.g., -1e8)
				return int((imt * q + (1. - imt) * -1e8).argmax(1))
		else:
			return np.random.randint(self.num_actions)


	def train(self, replay_buffer):
		# Sample replay buffer
		state, action, next_state, reward, done = replay_buffer.sample()

		# Get current Q estimate
		# use random ensemble Q-values during optimization
		current_Q, train_imt, train_i, std_matrix = self.Q(state, need_std=True)
		current_Q = current_Q.gather(1, action)

		# Compute the target Q value
		with torch.no_grad():
			# use average Q-values during no_grad
			# how to use std (without gradient) TODO
			q, imt, i, _ = self.Q.compute_Q(next_state, need_std=False)
			imt = imt.exp()
			imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()
			# action(replaybuffer) std
			action_std = std_matrix.gather(1, action)

			# Use large negative number to mask actions from argmax
			# next actions are selected by main Q
			next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)

			# target network use average Q-values
			q, imt, i, _ = self.Q_target.compute_Q(next_state, need_std=False)
			target_Q = (reward - action_std) + done * self.discount * q.gather(1, next_action).reshape(-1, 1)

		# Compute Q loss
		q_loss = F.smooth_l1_loss(current_Q, target_Q)
		i_loss = F.nll_loss(train_imt, action.reshape(-1))

		# i is logits
		Q_loss = q_loss + i_loss + 1e-2 * train_i.pow(2).mean()

		# Optimize the Q
		self.Q_optimizer.zero_grad()
		Q_loss.backward()
		self.Q_optimizer.step()

		# Update target network by polyak or full copy every X iterations.
		self.iterations += 1
		self.maybe_update_target()
		return Q_loss.mean().item()


	def polyak_target_update(self):
		for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
		   target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def copy_target_update(self):
		if self.iterations % self.target_update_frequency == 0:
			 self.Q_target.load_state_dict(self.Q.state_dict())