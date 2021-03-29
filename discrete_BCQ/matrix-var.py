import torch

a1 = torch.tensor([
	[1, 2, 3.],
	[2, 2, 2.],
	])

a2 = torch.tensor([
	[5, 6, 7.],
	[9, 9, 9.],
	])

matrix = torch.stack([a1, a2])
print("matrix.shape: ", matrix.shape)		# 2, 2, 3
# print(matrix[0])

print("-------")
print(matrix[:, 0, 0].shape, "\n", matrix[:, 0, 0])

std = []

for i in range(0, matrix.shape[1]):			# No.i sample
	action_std = []
	for j in range(0, matrix.shape[2]): 	# No.j action
		print("i sample, j action -> std & var: ", matrix[:, i, j].std(), matrix[:, i, j].var())
		action_std.append(matrix[:, i, j].std())
	std.append(action_std)

std = torch.tensor(std)
print(std.shape)
print("std: ", std)