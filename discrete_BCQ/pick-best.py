import torch

state = torch.tensor([
	[0, 0, 0.],
	[1, 1, 1],
	[2, 2, 2],
	[3, 3, 3],
	[4, 4, 4],
	])

best_idx = torch.LongTensor([0, 2, 4])

best_state = torch.index_select(state, dim=0, index=best_idx)
print(state)
print(state.shape)

print(best_state)
print(best_state.shape)

action_std = torch.tensor([1, 2, 3, 4, 5.]).reshape(-1, 1)
print(action_std)
print(action_std >= 3)
index, _ = torch.where(action_std >= 3)
best_state = torch.index_select(state, dim=0, index=index)
print(best_state)
print(best_state.shape)