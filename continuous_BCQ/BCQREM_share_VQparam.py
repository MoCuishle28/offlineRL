import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
BCQ's part from https://github.com/sfujim/BCQ
'''


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, phi=0.05):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 32)
		self.l2 = nn.Linear(32, 16)
		self.l3 = nn.Linear(16, action_dim)
		
		self.max_action = max_action
		self.phi = phi


	def forward(self, state, action):
		a = F.relu(self.l1(torch.cat([state, action], 1)))
		a = F.relu(self.l2(a))
		a = self.phi * self.max_action * torch.tanh(self.l3(a))
		return (a + action).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
		super(Critic, self).__init__()
		self.device = device
		# head 1
		# q1 state representation: l1, l2
		self.l1 = nn.Linear(state_dim + action_dim, 64)
		self.l2 = nn.Linear(64, 64)
		self.l3_1 = nn.Linear(64, 1)
		# q2 state representation: l4, l5
		self.l4 = nn.Linear(state_dim + action_dim, 64)
		self.l5 = nn.Linear(64, 64)
		self.l6_1 = nn.Linear(64, 1)

		# head 2
		self.l3_2 = nn.Linear(64, 1)	# q1
		self.l6_2 = nn.Linear(64, 1)	# q2

		# head 3
		self.l3_3 = nn.Linear(64, 1)	# q1
		self.l6_3 = nn.Linear(64, 1)	# q2

		# head 4
		self.l3_4 = nn.Linear(64, 1)	# q1
		self.l6_4 = nn.Linear(64, 1)	# q2

		# head 5
		self.l3_5 = nn.Linear(64, 1)	# q1
		self.l6_5 = nn.Linear(64, 1)	# q2

		# Vanilla Variational Auto-Encoder 
		self.mean = nn.Linear(64, latent_dim)
		self.log_std = nn.Linear(64, latent_dim)

		self.d1 = nn.Linear(state_dim + latent_dim, 64)
		self.d2 = nn.Linear(64, 64)
		self.d3 = nn.Linear(64, action_dim)

		self.max_action = max_action
		self.latent_dim = latent_dim
		self.device = device

	def compute_q(self, sa1, sa2, l3, l6):
		q1 = l3(sa1)
		q2 = l6(sa2)
		return q1, q2

	def compute_q1(self, sa1, l3):
		q1 = l3(sa1)
		return q1

	def forward(self, state, action):
		sa1 = F.relu(self.l1(torch.cat([state, action], 1)))
		sa1 = F.relu(self.l2(sa1))	# (state, action) pair latent vector

		sa2 = F.relu(self.l4(torch.cat([state, action], 1)))
		sa2 = F.relu(self.l5(sa2))		

		# head 1
		q1_1, q2_1 = self.compute_q(sa1, sa2, self.l3_1, self.l6_1)
		# head 2
		q1_2, q2_2 = self.compute_q(sa1, sa2, self.l3_2, self.l6_2)
		# head 3
		q1_3, q2_3 = self.compute_q(sa1, sa2, self.l3_3, self.l6_3)
		# head 4
		q1_4, q2_4 = self.compute_q(sa1, sa2, self.l3_4, self.l6_4)
		# head 5
		q1_5, q2_5 = self.compute_q(sa1, sa2, self.l3_5, self.l6_5)

		# random ensemble mixture
		alpha = torch.Tensor(5).uniform_(0,1).to(self.device)
		d = alpha.sum()
		alpha1, alpha2, alpha3, alpha4, aplha5 = torch.tensor([x/d for x in alpha]).to(self.device)

		q1 = alpha1*q1_1 + alpha2*q1_2 + alpha3*q1_3 + alpha4*q1_4 + aplha5*q1_5
		q2 = alpha1*q2_1 + alpha2*q2_2 + alpha3*q2_3 + alpha4*q2_4 + aplha5*q2_5

		# VAE 
		# 1. average state action pair vector
		sa = (sa1 + sa2) / 2
		# 2. sa1
		# sa = sa1 (state action pair vector)

		mean = self.mean(sa)
		# Clamped for numerical stability 
		log_std = self.log_std(sa).clamp(-4, 15)
		std = torch.exp(log_std)
		z = mean + std * torch.randn_like(std)		
		u = self.decode(state, z)

		return q1, q2, u, mean, std


	def q1(self, state, action):
		sa1 = F.relu(self.l1(torch.cat([state, action], 1)))
		sa1 = F.relu(self.l2(sa1))
		# head 1
		q1_1 = self.compute_q1(sa1, self.l3_1)
		# head 2
		q1_2 = self.compute_q1(sa1, self.l3_2)
		# head 3
		q1_3 = self.compute_q1(sa1, self.l3_3)
		# head 4
		q1_4 = self.compute_q1(sa1, self.l3_4)
		# head 5
		q1_5 = self.compute_q1(sa1, self.l3_5)

		q1 = ((q1_1 + q1_2 + q1_3 + q1_4 + q1_5) / 5.0)
		return q1

	def decode(self, state, z=None):
		# When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
		if z is None:
			z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

		a = F.relu(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))
		return self.max_action * torch.tanh(self.d3(a))	


class BCQ(object):
	def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
		latent_dim = action_dim * 2

		self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

		self.critic = Critic(state_dim, action_dim, latent_dim, max_action, device).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

		self.max_action = max_action
		self.action_dim = action_dim
		self.discount = discount
		self.tau = tau
		self.lmbda = lmbda
		self.device = device


	def select_action(self, state):		
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
			action = self.actor(state, self.critic.decode(state))
			q1 = self.critic.q1(state, action)
			ind = q1.argmax(0)
		return action[ind].cpu().data.numpy().flatten()


	def train(self, replay_buffer, iterations, batch_size=100):

		for it in range(iterations):
			# Sample replay buffer / batch
			state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

			# Variational Auto-Encoder Training
			current_Q1, current_Q2, recon, mean, std = self.critic(state, action)
			recon_loss = F.mse_loss(recon, action)
			KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
			vae_loss = recon_loss + 0.5 * KL_loss

			# Critic Training
			with torch.no_grad():
				# Duplicate next state 10 times
				next_state = torch.repeat_interleave(next_state, 10, 0)

				# Compute value of perturbed actions sampled from the VAE
				target_Q1, target_Q2, _, _, _ = self.critic_target(next_state, self.actor_target(next_state, self.critic.decode(next_state)))

				# Soft Clipped Double Q-learning 
				target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
				# Take max over each action sampled from the VAE
				target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

				target_Q = reward + not_done * self.discount * target_Q

			# current_Q1, current_Q2, _, _, _ = self.critic(state, action)
			# original loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			# smooth_l1_loss (Huber Loss)
			# critic_loss = F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)

			share_critic_loss = critic_loss + vae_loss
			self.critic_optimizer.zero_grad()
			share_critic_loss.backward()
			self.critic_optimizer.step()


			# Pertubation Model / Action Training
			sampled_actions = self.critic.decode(state)
			perturbed_actions = self.actor(state, sampled_actions)

			# Update through DPG
			actor_loss = -self.critic.q1(state, perturbed_actions).mean()
		 	 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()


			# Update Target Networks 
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		return vae_loss, actor_loss, critic_loss