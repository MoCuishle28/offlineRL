import numpy as np

seed = input('seed:')
# look results(average rewards)
# setting = f'PongNoFrameskip-v0_{seed}.npy'
setting = f'AmidarNoFrameskip-v0_{seed}.npy'

print('SETTING:', setting)

# DQN
try:
	results = np.load(f"./results/behavioral_{setting}")
	print('behavioral(DQN):', results)
except Exception as e:
	print('no behavioral results...')

# Replay Buffer
results = np.load(f"./results/buffer_performance_{setting}")
print('Replay Buffer:', results)

# BCQ
results = np.load(f"./results/BCQ_{setting}")
print('BCQ:', results)

# # BCQREM
results = np.load(f"./results/BCQREM_{setting}")
print('BCQREM:', results)

# one noise
results = np.load(f"./results/BCQREMoneimt_{setting}")
print('BCQREMoneimt:', results)

# multi imt head noise
results = np.load(f"./results/BCQREMmultiimt_{setting}")
print('BCQREMmultiimt:', results)
