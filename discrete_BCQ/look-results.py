import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--env", default='Pong')	# AmidarNoFrameskip-v0
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model", default="sl")
args = parser.parse_args()

# look results(average rewards)
# setting = f'PongNoFrameskip-v0_{seed}.npy'
# setting = f'AmidarNoFrameskip-v0_{seed}.npy'
setting = f'{args.env}NoFrameskip-v0_{args.seed}.npy'

print('SETTING:', setting)

base_dir = 'results/'
path = f"{base_dir}{args.model}_{setting}"
results = np.load(path)
print(path)
print('results:', results, results.shape)

# # DQN
# try:
# 	results = np.load(f"./results/behavioral_{setting}")
# 	print('behavioral(DQN):', results)
# except Exception as e:
# 	print('no behavioral results...')

# # Replay Buffer
# results = np.load(f"./results/buffer_performance_{setting}")
# print('Replay Buffer:', results)

# # BCQ
# results = np.load(f"./results/BCQ_{setting}")
# print('BCQ:', results)

# # # BCQREM
# results = np.load(f"./results/BCQREM_{setting}")
# print('BCQREM:', results)

# # one noise
# results = np.load(f"./results/BCQREMoneimt_{setting}")
# print('BCQREMoneimt:', results)

# # multi imt head noise
# results = np.load(f"./results/BCQREMmultiimt_{setting}")
# print('BCQREMmultiimt:', results)
