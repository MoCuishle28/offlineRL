import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Used for Atari
class Conv(nn.Module):
	def __init__(self, frames, num_actions, device):
		super(Conv, self).__init__()
		self.device = device

		self.c1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
		self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
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

	def compute(self, state):
		c = F.relu(self.c1(state))
		c = F.relu(self.c2(c))
		c = F.relu(self.c3(c))

		i = F.relu(self.i1(c.reshape(-1, 3136)))
		i_1, i_2, i_3, i_4, i_5 = self.i2_1(i), self.i2_2(i), self.i2_3(i), self.i2_4(i), self.i2_5(i)
		i = (i_1 + i_2 + i_3 + i_4 + i_5) / 5.0
		return F.log_softmax(i, dim=1), i


	def forward(self, state):
		s = F.relu(self.c1(state))
		s = F.relu(self.c2(s))
		s = F.relu(self.c3(s))
		i = F.relu(self.i1(s.reshape(-1, 3136)))

		# compute ensemble logits (batch, action dim)
		i_1, i_2, i_3, i_4, i_5 = self.i2_1(i), self.i2_2(i), self.i2_3(i), self.i2_4(i), self.i2_5(i)

		# random ensemble mixture
		batch = i.shape[0]
		alpha = torch.Tensor(batch, 5).uniform_(0, 1).to(self.device)
		d = alpha.sum(dim=1)
		alpha = torch.stack([x/d[idx] for idx, x in enumerate(alpha)]).to(self.device)

		# random ensemble logits
		i = alpha[:, 0].view(-1, 1)*i_1 + alpha[:, 1].view(-1, 1)*i_2 + alpha[:, 2].view(-1, 1)*i_3 + alpha[:, 3].view(-1, 1)*i_4 + alpha[:, 4].view(-1, 1)*i_5
		return F.log_softmax(i, dim=1), i


# Used for Box2D / Toy problems
class FC(nn.Module):
	def __init__(self, state_dim, num_actions):
		super(FC, self).__init__()
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, num_actions)


	def forward(self, state):
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(q))
		return self.l3(q)


class Model(object):
	def __init__(
		self, 
		is_atari,
		num_actions,
		state_dim,
		device,
		optimizer="Adam",
		optimizer_parameters={},
	):
	
		self.device = device

		# Determine network type
		self.model = Conv(state_dim[0], num_actions, device).to(self.device) if is_atari else FC(state_dim, num_actions).to(self.device)
		self.optimizer = getattr(torch.optim, optimizer)(self.model.parameters(), **optimizer_parameters)

		self.state_shape = (-1,) + state_dim if is_atari else (-1, state_dim)
		self.num_actions = num_actions


	def select_action(self, state, eval=False):
		with torch.no_grad():
			state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
			imt, i = self.model.compute(state)
			return int(imt.argmax(1))


	def train(self, replay_buffer):
		# Sample replay buffer
		state, action, next_state, reward, done = replay_buffer.sample()

		imt, _ = self.model(state)

		# Compute supervised learning loss
		i_loss = F.nll_loss(imt, action.reshape(-1))

		# Optimize the model
		self.optimizer.zero_grad()
		i_loss.backward()
		self.optimizer.step()

		return i_loss.item()