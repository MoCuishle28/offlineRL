import argparse
import copy
import importlib
import json
import os

import numpy as np
import torch

import discrete_BCQREM
import discrete_BCQREM_one_imt_noise
import discrete_BCQREM_multi_imt_noise
import discrete_BCQREM_no_batch_alpha
import discrete_BCQREM_softmax
import discrete_BCQREM_one_condition_noise
import BCQREM_reward_var
import BCQREM_cond_var
import BCQREM_adw_reward_var
import BCQREM_adw_cond_var
import BCQREM_adw

import discrete_BCQ
import DQN
import utils


def interact_with_environment(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters):
	# For saving files
	setting = f"{args.env}_{args.seed}"
	buffer_name = f"{args.buffer_name}_{setting}"

	# Initialize and load policy
	policy = DQN.DQN(
		is_atari,
		num_actions,
		state_dim,
		device,
		parameters["discount"],
		parameters["optimizer"],
		parameters["optimizer_parameters"],
		parameters["polyak_target_update"],
		parameters["target_update_freq"],
		parameters["tau"],
		parameters["initial_eps"],
		parameters["end_eps"],
		parameters["eps_decay_period"],
		parameters["eval_eps"],
	)

	if args.generate_buffer: policy.load(f"./models/behavioral_{setting}")
	
	evaluations = []

	state, done = env.reset(), False
	episode_start = True
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	low_noise_ep = np.random.uniform(0,1) < args.low_noise_p

	# Interact with the environment for max_timesteps
	for t in range(int(args.max_timesteps)):

		episode_timesteps += 1

		# If generating the buffer, episode is low noise with p=low_noise_p.
		# If policy is low noise, we take random actions with p=eval_eps.
		# If the policy is high noise, we take random actions with p=rand_action_p.
		if args.generate_buffer:
			if not low_noise_ep and np.random.uniform(0,1) < args.rand_action_p - parameters["eval_eps"]:
				action = env.action_space.sample()
			else:
				action = policy.select_action(np.array(state), eval=True)

		if args.train_behavioral:
			if t < parameters["start_timesteps"]:
				action = env.action_space.sample()
			else:
				action = policy.select_action(np.array(state))

		# Perform action and log results
		next_state, reward, done, info = env.step(action)
		episode_reward += reward

		# Only consider "done" if episode terminates due to failure condition
		done_float = float(done) if episode_timesteps < env._max_episode_steps else 0

		# For atari, info[0] = clipped reward, info[1] = done_float
		if is_atari:
			reward = info[0]
			done_float = info[1]
			
		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_float, done, episode_start)
		state = copy.copy(next_state)
		episode_start = False

		# Train agent after collecting sufficient data
		if args.train_behavioral and t >= parameters["start_timesteps"] and (t+1) % parameters["train_freq"] == 0:
			policy.train(replay_buffer)

		if done:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1}/[{int(args.max_timesteps)}] Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False
			episode_start = True
			episode_reward = 0
			episode_timesteps = 0	# interaction times in one game episode?
			episode_num += 1	# game over times
			low_noise_ep = np.random.uniform(0,1) < args.low_noise_p

		# Evaluate episode
		if args.train_behavioral and (t + 1) % parameters["eval_freq"] == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(f"./results/behavioral_{setting}", evaluations)
			policy.save(f"./models/behavioral_{setting}")

	# Save final policy
	if args.train_behavioral:
		policy.save(f"./models/behavioral_{setting}")

	# Save final buffer and performance
	else:
		evaluations.append(eval_policy(policy, args.env, args.seed))
		np.save(f"./results/buffer_performance_{setting}", evaluations)
		replay_buffer.save(f"./buffers/{buffer_name}")


# Trains BCQ-REM offline
def train_BCQREM(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters):
	# For saving files
	setting = f"{args.env}_{args.seed}"
	buffer_name = f"{args.buffer_name}_{setting}"

	# Initialize and load policy
	if args.model == 'BCQ':
		print('creating BCQ')
		policy = discrete_BCQ.discrete_BCQ(
			is_atari,
			num_actions,
			state_dim,
			device,
			args.BCQ_threshold,
			parameters["discount"],
			parameters["optimizer"],
			parameters["optimizer_parameters"],
			parameters["polyak_target_update"],
			parameters["target_update_freq"],
			parameters["tau"],
			parameters["initial_eps"],
			parameters["end_eps"],
			parameters["eps_decay_period"],
			parameters["eval_eps"]
		)
	elif args.model == 'BCQREM': 	# creating BCQ-REM
		print('creating BCQ-REM')
		policy = discrete_BCQREM.discrete_BCQ(
			is_atari,
			num_actions,
			state_dim,
			device,
			args.BCQ_threshold,
			parameters["discount"],
			parameters["optimizer"],
			parameters["optimizer_parameters"],
			parameters["polyak_target_update"],
			parameters["target_update_freq"],
			parameters["tau"],
			parameters["initial_eps"],
			parameters["end_eps"],
			parameters["eps_decay_period"],
			parameters["eval_eps"]
		)
	elif args.model == 'BCQREMoneimt':
		print('creating BCQ-REM with one imt head')
		policy = discrete_BCQREM_one_imt_noise.discrete_BCQ(
			is_atari,
			num_actions,
			state_dim,
			device,
			args.BCQ_threshold,
			parameters["discount"],
			parameters["optimizer"],
			parameters["optimizer_parameters"],
			parameters["polyak_target_update"],
			parameters["target_update_freq"],
			parameters["tau"],
			parameters["initial_eps"],
			parameters["end_eps"],
			parameters["eps_decay_period"],
			parameters["eval_eps"]
		)
	elif args.model == 'BCQREMmultiimt':
		print('creating BCQ-REM with multi imt head')
		policy = discrete_BCQREM_multi_imt_noise.discrete_BCQ(
			is_atari,
			num_actions,
			state_dim,
			device,
			args.BCQ_threshold,
			parameters["discount"],
			parameters["optimizer"],
			parameters["optimizer_parameters"],
			parameters["polyak_target_update"],
			parameters["target_update_freq"],
			parameters["tau"],
			parameters["initial_eps"],
			parameters["end_eps"],
			parameters["eps_decay_period"],
			parameters["eval_eps"]
		)
	elif args.model == 'BCQREMnobatch':
		print('creating BCQ-REM (alpha without batch)')
		policy = discrete_BCQREM_no_batch_alpha.discrete_BCQ(
			is_atari,
			num_actions,
			state_dim,
			device,
			args.BCQ_threshold,
			parameters["discount"],
			parameters["optimizer"],
			parameters["optimizer_parameters"],
			parameters["polyak_target_update"],
			parameters["target_update_freq"],
			parameters["tau"],
			parameters["initial_eps"],
			parameters["end_eps"],
			parameters["eps_decay_period"],
			parameters["eval_eps"]
		)
	elif args.model == 'BCQREMsoftmax':
		print('creating BCQ-REM (softmax alpha)')
		policy = discrete_BCQREM_softmax.discrete_BCQ(
			is_atari,
			num_actions,
			state_dim,
			device,
			args.BCQ_threshold,
			parameters["discount"],
			parameters["optimizer"],
			parameters["optimizer_parameters"],
			parameters["polyak_target_update"],
			parameters["target_update_freq"],
			parameters["tau"],
			parameters["initial_eps"],
			parameters["end_eps"],
			parameters["eps_decay_period"],
			parameters["eval_eps"]
		)
	elif args.model == 'BCQREMcon':
		print('creating BCQ-REM with one supervised head (only condition noise)')
		policy = discrete_BCQREM_one_condition_noise.discrete_BCQ(
			is_atari,
			num_actions,
			state_dim,
			device,
			args.BCQ_threshold,
			parameters["discount"],
			parameters["optimizer"],
			parameters["optimizer_parameters"],
			parameters["polyak_target_update"],
			parameters["target_update_freq"],
			parameters["tau"],
			parameters["initial_eps"],
			parameters["end_eps"],
			parameters["eps_decay_period"],
			parameters["eval_eps"]
		)
	elif args.model == 'BCQREMrvar':
		print('creating BCQ-REM with multi supervised head (without noise), reward - std')
		policy = BCQREM_reward_var.discrete_BCQ(
			is_atari,
			num_actions,
			state_dim,
			device,
			args.BCQ_threshold,
			parameters["discount"],
			parameters["optimizer"],
			parameters["optimizer_parameters"],
			parameters["polyak_target_update"],
			parameters["target_update_freq"],
			parameters["tau"],
			parameters["initial_eps"],
			parameters["end_eps"],
			parameters["eps_decay_period"],
			parameters["eval_eps"],
		)
	elif args.model == 'BCQREMcondvar':
		print('creating BCQ-REM with multi supervised head (without noise), probability / var')
		policy = BCQREM_cond_var.discrete_BCQ(
			is_atari,
			num_actions,
			state_dim,
			device,
			args.BCQ_threshold,
			parameters["discount"],
			parameters["optimizer"],
			parameters["optimizer_parameters"],
			parameters["polyak_target_update"],
			parameters["target_update_freq"],
			parameters["tau"],
			parameters["initial_eps"],
			parameters["end_eps"],
			parameters["eps_decay_period"],
			parameters["eval_eps"],
		)
	elif args.model == 'BCQREMadwrvar':
		print('creating BCQ-REM with multi supervised head (without noise), reward - std, adw SL model')
		policy = BCQREM_adw_reward_var.discrete_BCQ(
			is_atari,
			num_actions,
			state_dim,
			device,
			args.BCQ_threshold,
			parameters["discount"],
			parameters["optimizer"],
			parameters["optimizer_parameters"],
			parameters["polyak_target_update"],
			parameters["target_update_freq"],
			parameters["tau"],
			parameters["initial_eps"],
			parameters["end_eps"],
			parameters["eps_decay_period"],
			parameters["eval_eps"],
		)
	elif args.model == 'BCQREMadwcondvar':
		print('creating BCQ-REM with multi supervised head (without noise), probability / var, adw SL model')
		policy = BCQREM_adw_cond_var.discrete_BCQ(
			is_atari,
			num_actions,
			state_dim,
			device,
			args.BCQ_threshold,
			parameters["discount"],
			parameters["optimizer"],
			parameters["optimizer_parameters"],
			parameters["polyak_target_update"],
			parameters["target_update_freq"],
			parameters["tau"],
			parameters["initial_eps"],
			parameters["end_eps"],
			parameters["eps_decay_period"],
			parameters["eval_eps"],
		)
	elif args.model == 'BCQREMadw':
		print('creating BCQ-REM with multi supervised head (without noise), adw SL model')
		policy = BCQREM_adw.discrete_BCQ(
			is_atari,
			num_actions,
			state_dim,
			device,
			args.BCQ_threshold,
			parameters["discount"],
			parameters["optimizer"],
			parameters["optimizer_parameters"],
			parameters["polyak_target_update"],
			parameters["target_update_freq"],
			parameters["tau"],
			parameters["initial_eps"],
			parameters["end_eps"],
			parameters["eps_decay_period"],
			parameters["eval_eps"],
		)


	# Load replay buffer	
	replay_buffer.load(f"./buffers/{buffer_name}")
	
	evaluations = []
	episode_num = 0
	done = True 
	training_iters = 0

	print('NO.1 evaluations...')
	evaluations.append(eval_policy(policy, args.env, args.seed))	# TODO DEBUG
	while training_iters < args.max_timesteps: 
		
		for train_policy_times in range(int(parameters["eval_freq"])):
			mean_loss = policy.train(replay_buffer)
			if train_policy_times % (int(parameters["eval_freq"]) // 5) == 0:
				print(f'times:{train_policy_times}/[{int(parameters["eval_freq"])}], Mean Loss: {round(mean_loss, 5)}')

		evaluations.append(eval_policy(policy, args.env, args.seed))
		if args.model == 'BCQ':
			np.save(f"./results/BCQ_{setting}", evaluations)	# TODO
		elif args.model == 'BCQREM':
			np.save(f"./results/BCQREM_{setting}", evaluations)	# TODO
		elif args.model == 'BCQREMoneimt':
			np.save(f"./results/BCQREMoneimt_{setting}", evaluations)
		elif args.model == 'BCQREMmultiimt':
			np.save(f"./results/BCQREMmultiimt_{setting}", evaluations)
		elif args.model == 'BCQREMnobatch':
			np.save(f"./results/BCQREMnobatch_{setting}", evaluations)
		elif args.model == 'BCQREMsoftmax':
			np.save(f"./results/BCQREMsoftmax_{setting}", evaluations)
		elif args.model == 'BCQREMcon':
			np.save(f"./results/BCQREMcon_{setting}", evaluations)
		elif args.model == 'BCQREMrvar':
			np.save(f"./results/BCQREMrvar_{setting}", evaluations)
		elif args.model == 'BCQREMcondvar':
			np.save(f"./results/BCQREMcondvar_{setting}", evaluations)
		elif args.model == 'BCQREMadwrvar':
			np.save(f"./results/BCQREMadwrvar_{setting}", evaluations)
		elif args.model == 'BCQREMadwcondvar':
			np.save(f"./results/BCQREMadwcondvar_{setting}", evaluations)
		elif args.model == 'BCQREMadw':
			np.save(f"./results/BCQREMadw_{setting}", evaluations)

		training_iters += int(parameters["eval_freq"])
		print(f"Training iterations: {training_iters}/[{int(args.max_timesteps)}]")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env, _, _, _ = utils.make_env(env_name, atari_preprocessing)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state), eval=True)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":

	# Atari Specific
	atari_preprocessing = {
		"frame_skip": 4,
		"frame_size": 84,
		"state_history": 4,
		"done_on_life_loss": False,
		"reward_clipping": True,
		"max_episode_timesteps": 27e3
	}

	atari_parameters = {
		# Exploration
		"start_timesteps": 2e4,	# start train behavioral policy
		"initial_eps": 1,
		"end_eps": 1e-2,
		"eps_decay_period": 25e4,
		# Evaluation
		"eval_freq": 5e4,
		"eval_eps": 1e-3,	# for explore
		# Learning
		"discount": 0.99,
		"buffer_size": 1e6,
		"batch_size": 32,
		"optimizer": "Adam",
		"optimizer_parameters": {
			"lr": 0.0000625,
			"eps": 0.00015
		},
		"train_freq": 4,		# training frequence
		"polyak_target_update": False,
		"target_update_freq": 8e3,
		"tau": 0.005	# original: 1
	}

	regular_parameters = {
		# Exploration
		"start_timesteps": 1e3,
		"initial_eps": 0.1,
		"end_eps": 0.1,
		"eps_decay_period": 1,
		# Evaluation
		"eval_freq": 5e3,
		"eval_eps": 0,
		# Learning
		"discount": 0.99,
		"buffer_size": 1e6,
		"batch_size": 64,
		"optimizer": "Adam",
		"optimizer_parameters": {
			"lr": 3e-4
		},
		"train_freq": 1,
		"polyak_target_update": True,
		"target_update_freq": 1,
		"tau": 0.005
	}

	# Load parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="PongNoFrameskip-v0")     # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_name", default="Default")        # Prepends name to filename
	parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment or train for
	parser.add_argument("--BCQ_threshold", default=0.3, type=float)# Threshold hyper-parameter for BCQ
	parser.add_argument("--low_noise_p", default=0.2, type=float)  # Probability of a low noise episode when generating buffer
	parser.add_argument("--rand_action_p", default=0.2, type=float)# Probability of taking a random action when generating buffer, during non-low noise episode
	parser.add_argument("--train_behavioral", action="store_true") # If true, train behavioral policy
	parser.add_argument("--generate_buffer", action="store_true")  # If true, generate buffer
	# parser.add_argument("--BCQ", action="store_true")  	# If true, train original BCQ

	'''
	choose model: 
		BCQ, BCQREM, BCQREMoneimt(noise), BCQREMmultiimt, BCQREMnobatch(no batch alpha)
		BCQREMsoftmax (softmax alpha), BCQREMcon (only condition noise) (one supervised head), 
		BCQREMrvar (without noise, reward - std), 
		BCQREMcondvar (cond: probability / var),
		BCQREMadwrvar (adw SL model, reward - std),
		BCQREMadwcondvar (adw SL model, cond: probability / var)
		BCQREMadw
	'''
	parser.add_argument("--model", default='BCQ')
	parser.add_argument('--polyak', default='n')	# y / n -> polyak_target_update / NO polyak_target_update
	parser.add_argument("--var_threshold", default=0.3, type=float)#  Threshold for action var
	args = parser.parse_args()
	
	print("---------------------------------------")	
	if args.train_behavioral:
		print(f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}")
	elif args.generate_buffer:
		print(f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}")
	else:
		print(f"Setting: Training {args.model}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if args.train_behavioral and args.generate_buffer:
		print("Train_behavioral and generate_buffer cannot both be true.")
		exit()

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if not os.path.exists("./models"):
		os.makedirs("./models")

	if not os.path.exists("./buffers"):
		os.makedirs("./buffers")

	# Make env and determine properties
	env, is_atari, state_dim, num_actions = utils.make_env(args.env, atari_preprocessing)
	parameters = atari_parameters if is_atari else regular_parameters
	parameters['polyak_target_update'] = True if args.polyak == 'y' else False
	print(parameters)

	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Initialize buffer
	replay_buffer = utils.ReplayBuffer(state_dim, is_atari, atari_preprocessing, parameters["batch_size"], parameters["buffer_size"], device)

	if args.train_behavioral or args.generate_buffer:
		interact_with_environment(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters)
	else:
		train_BCQREM(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters)