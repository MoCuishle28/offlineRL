import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--env", default='Pong')	# AmidarNoFrameskip-v0
parser.add_argument("--e", type=int, default=10)
args = parser.parse_args()


seeds = [0, 2021, 9527]
if args.env == 'Amidar':
	seeds = [10, 20, 30]

tail = 'NoFrameskip-v0'
args.env = args.env + tail
base_dir = 'results/'
total_eval_times = 21

x, i = [], 5e4 * (21 - args.e)
for _ in range(args.e):
	x.append(i)
	i += 5e4

bcqrem_adw_var_term1_res_list, bcqrem_adw_var_term2_res_list = [], []

bcqrem_adw_var_weight_QLoss_res_list = []
bcqrem_adw_var_weight_res_list = []

bcqrem_adw_A_weight_res_list = []

bcq_res_list = []
sl_res_list = []
multisl_res_list = []
rem_res_list = []

bcq_one_imt_adw_res_list = []
bcq_multi_imt_adw_res_list = []
bcqrem_one_adw_res_list = []
bcqrem_adw_res_list = []

bcq_multi_imt_adw_cond_var_res_list = []
bcq_2Q_multi_imt_adw_cond_var_res_list = []		# aver Q
bcq_2Q_multi_imt_adw_cond_var_res_list2 = []	# choose Q1

for seed in seeds:
	setting = f'{args.env}_{seed}.npy'

	# bcqrem_adw_var_term1_res = np.load(f"{base_dir}BCQREMadwvart1_1.0_{setting}")
	# bcqrem_adw_var_term1_res_list.append(bcqrem_adw_var_term1_res[total_eval_times - args.e : ])

	bcqrem_adw_var_term2_res = np.load(f"{base_dir}BCQREMadwvart2_1.0_{setting}")	# minus action_var version
	bcqrem_adw_var_term2_res_list.append(bcqrem_adw_var_term2_res[total_eval_times - args.e : ])

	# ------------------ baseline ------------------
	sl_res = np.load(f"{base_dir}/sl_{setting}")
	sl_res_list.append(sl_res[total_eval_times - args.e : ])

	multisl_res = np.load(f"{base_dir}/slmulti_{setting}")
	multisl_res_list.append(multisl_res[total_eval_times - args.e : ])
	
	rem_res = np.load(f"{base_dir}/rem_{setting}")
	rem_res_list.append(rem_res[total_eval_times - args.e : ])

	bcq_res = np.load(f"{base_dir}/BCQ_{setting}")
	bcq_res_list.append(bcq_res[total_eval_times - args.e : ])

	bcq_one_imt_adw_res = np.load(f"{base_dir}/BCQadw_{setting}")		# BCQ adw one imt
	bcq_one_imt_adw_res_list.append(bcq_one_imt_adw_res[total_eval_times - args.e : ])

	bcq_multi_imt_adw_res = np.load(f"{base_dir}/BCQmultiimtadw_{setting}")		# BCQ adw multi imt
	bcq_multi_imt_adw_res_list.append(bcq_multi_imt_adw_res[total_eval_times - args.e : ])

	bcqrem_one_adw_res = np.load(f"{base_dir}/BCQREMadwone_{setting}")		# BCQREM adw one imt
	bcqrem_one_adw_res_list.append(bcqrem_one_adw_res[total_eval_times - args.e : ])

	bcqrem_adw_res = np.load(f"{base_dir}/BCQREMadw_{setting}")		# BCQREM adw multi imt
	bcqrem_adw_res_list.append(bcqrem_adw_res[total_eval_times - args.e : ])
	# ----------------------------------------------

	bcqrem_adw_A_weight_res = np.load(f"{base_dir}/BCQREMadwAw_{setting}")
	bcqrem_adw_A_weight_res_list.append(bcqrem_adw_A_weight_res[total_eval_times - args.e : ])

	# BCQ Multi imt adw cond-var
	bcq_multi_imt_adw_cond_var_res = np.load(f"{base_dir}/BCQmultiimtadwcondvar_{setting}")
	bcq_multi_imt_adw_cond_var_res_list.append(bcq_multi_imt_adw_cond_var_res[total_eval_times - args.e : ])

	# BCQ 2Q Multi imt adw cond-var (aver Q in eval)
	bcq_2Q_multi_imt_adw_cond_var_res = np.load(f"{base_dir}/BCQ2Qmultiimtadwcondvar-averQ-in-eval/BCQ2Qmultiimtadwcondvar_{setting}")
	bcq_2Q_multi_imt_adw_cond_var_res_list.append(bcq_2Q_multi_imt_adw_cond_var_res[total_eval_times - args.e : ])

	# BCQ 2Q Multi imt adw cond-var (choose Q1 in eval)
	bcq_2Q_multi_imt_adw_cond_var_res2 = np.load(f"{base_dir}/BCQ2Qmultiimtadwcondvar_{setting}")
	bcq_2Q_multi_imt_adw_cond_var_res_list2.append(bcq_2Q_multi_imt_adw_cond_var_res2[total_eval_times - args.e : ])


sl_res = sum(sl_res_list) / len(sl_res_list)
multisl_res = sum(multisl_res_list) / len(multisl_res_list)
rem_res = sum(rem_res) / len(rem_res)
bcq_res = sum(bcq_res_list) / len(bcq_res_list)

# --------- adw ----------
bcq_one_imt_adw_res = sum(bcq_one_imt_adw_res_list) / len(bcq_one_imt_adw_res_list)
bcq_multi_imt_adw_res = sum(bcq_multi_imt_adw_res_list) / len(bcq_multi_imt_adw_res_list)
bcqrem_one_adw_res = sum(bcqrem_one_adw_res_list) / len(bcqrem_one_adw_res_list)
bcqrem_adw_res = sum(bcqrem_adw_res_list) / len(bcqrem_adw_res_list)

# bcqrem_adw_var_term1_res = sum(bcqrem_adw_var_term1_res_list) / len(bcqrem_adw_var_term1_res_list)
bcqrem_adw_var_term2_res = sum(bcqrem_adw_var_term2_res_list) / len(bcqrem_adw_var_term2_res_list)

# var weight
bcqrem_adw_A_weight_res = sum(bcqrem_adw_A_weight_res_list) / len(bcqrem_adw_A_weight_res_list)

bcq_multi_imt_adw_cond_var_res = sum(bcq_multi_imt_adw_cond_var_res_list) / len(bcq_multi_imt_adw_cond_var_res_list)
bcq_2Q_multi_imt_adw_cond_var_res = sum(bcq_2Q_multi_imt_adw_cond_var_res_list) / len(bcq_2Q_multi_imt_adw_cond_var_res_list)
bcq_2Q_multi_imt_adw_cond_var_res2 = sum(bcq_2Q_multi_imt_adw_cond_var_res_list2) / len(bcq_2Q_multi_imt_adw_cond_var_res_list2)

print("ENV: ", args.env)
print("--------- baseline ----------")
print("SL:", f"Max: {round(sl_res.max().item(), 5)}", f"Aver: {round(sl_res.mean().item(), 5)}")
print("Multi SL:", f"Max: {round(multisl_res.max().item(), 5)}", f"Aver: {round(multisl_res.mean().item(), 5)}")
print("REM:", f"Max: {round(rem_res.max().item(), 5)}", f"Aver: {round(rem_res.mean().item(), 5)}")
print("BCQ:", f"Max: {round(bcq_res.max().item(), 5)}", f"Aver: {round(bcq_res.mean().item(), 5)}")

print("--------- adw ----------")
print("BCQ adw one imt:", f"Max: {round(bcq_one_imt_adw_res.max().item(), 5)}", f"Aver: {round(bcq_one_imt_adw_res.mean().item(), 5)}")
print("BCQ adw multi imt:", f"Max: {round(bcq_multi_imt_adw_res.max().item(), 5)}", f"Aver: {round(bcq_multi_imt_adw_res.mean().item(), 5)}")
print("BCQREM adw one imt:", f"Max: {round(bcqrem_one_adw_res.max().item(), 5)}", f"Aver: {round(bcqrem_one_adw_res.mean().item(), 5)}")
print("BCQREM adw multi imt:", f"Max: {round(bcqrem_adw_res.max().item(), 5)}", f"Aver: {round(bcqrem_adw_res.mean().item(), 5)}")


print("--------- consider var --------- ")
# print("BCQREM adw var term1:", f"Max: {round(bcqrem_adw_var_term1_res.max().item(), 5)}", f"Aver: {round(bcqrem_adw_var_term1_res.mean().item(), 5)}")
print("BCQREM adw var term2(var in final Loss, minus action_var):", f"Max: {round(bcqrem_adw_var_term2_res.max().item(), 5)}", f"Aver: {round(bcqrem_adw_var_term2_res.mean().item(), 5)}")

print("BCQREM adw A weighting:", f"Max: {round(bcqrem_adw_A_weight_res.max().item(), 5)}", f"Aver: {round(bcqrem_adw_A_weight_res.mean().item(), 5)}")

print("BCQ adw multi imt cond var:", f"Max: {round(bcq_multi_imt_adw_cond_var_res.max().item(), 5)}", f"Aver: {round(bcq_multi_imt_adw_cond_var_res.mean().item(), 5)}")
print("BCQ(choose min in 2 Q-head) adw multi imt cond var(aver Q in eval):", f"Max: {round(bcq_2Q_multi_imt_adw_cond_var_res.max().item(), 5)}", f"Aver: {round(bcq_2Q_multi_imt_adw_cond_var_res.mean().item(), 5)}")
print("BCQ(choose min in 2 Q-head) adw multi imt cond var(choose Q1 in eval):", f"Max: {round(bcq_2Q_multi_imt_adw_cond_var_res2.max().item(), 5)}", f"Aver: {round(bcq_2Q_multi_imt_adw_cond_var_res2.mean().item(), 5)}")