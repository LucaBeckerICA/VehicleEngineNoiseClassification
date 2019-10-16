import kraftfahrzeuge_final.siamese_nn as nn
import json
import numpy as np
import time

# Best Configuration so far: acc=0.9 -> Conf = [[18,2],[2,18]]
#epochs = 200
#learning = 0.0001
#b = 1
#f = [2, 1]
#k = [[3,3], [3,3]]
#p = [[2,2],[2,2]]
#d = [0, 0]

# Very best ca 95 % With Xi32
#epochs = 300
#learning = 0.0001
#b = 1
#f = [16, 1]
#k = [[5,5], [3,3]]
#p = [[2,2],[2,2]]
#d = [0, 0]



epochs = 15
learning = 0.00008
b = 1
f = [16, 8]
k = [[5,5], [3,3]]
p = [[2,2],[2,2]]
d = 0.8
u = 256

files1024 = ['modpcen_dieselbenzin_N1024fs16kHz_large_balanced.mat', 'modpcen_lkwpkw_N1024fs16kHz_balanced.mat',
             'modmfccs_dieselbenzin_N1024Xi32_large_balanced.mat', 'modmfccs_lkwpkw_N1024Xi32_balanced.mat']
files512 = ['modpcen_dieselbenzin_N512fs16kHz_large_balanced.mat', 'modpcen_lkwpkw_N512fs16kHz_balanced.mat']
files = ['modpcen_dieselbenzin_N256fs16kHz_large_balanced.mat', 'modpcen_lkwpkw_N256fs16kHz_balanced.mat',
         'modmfccs_dieselbenzin_N256Xi32_large_balanced.mat', 'modmfccs_lkwpkw_N256Xi32t2sekK4.mat']
# Now for N Crossvalidations:
cross_validations = 1
conf_mats = np.zeros((cross_validations, 2, 2))
mean_cvs = np.zeros((2, 2))
std_cvs = np.zeros((2, 2))
accs = np.zeros(cross_validations)
#means = np.zeros(1)
#stds = np.zeros(1)


start = time.time()
for i in range(cross_validations):
    print("Cross-Validation " + str(i + 1) + " of " + str(cross_validations))
    accs[i], conf_mats[i, :, :] = nn.train_nn(files1024[0], learning, epochs, [0.15, 0.2], b, f, k, p, d, u)

mean_cvs = np.mean(conf_mats, axis=0)
# Standard Deviations:
std11 = np.sqrt(np.sum(np.square(conf_mats[:, 0, 0] - mean_cvs[0, 0])) / cross_validations)
std12 = np.sqrt(np.sum(np.square(conf_mats[:, 0, 1] - mean_cvs[0, 1])) / cross_validations)
std22 = np.sqrt(np.sum(np.square(conf_mats[:, 1, 1] - mean_cvs[1, 1])) / cross_validations)
std21 = np.sqrt(np.sum(np.square(conf_mats[:, 1, 0] - mean_cvs[1, 0])) / cross_validations)
std_cvs[0, 0] = std11
std_cvs[0, 1] = std12
std_cvs[1, 1] = std22
std_cvs[1, 0] = std21
means = np.mean(accs[:])
stds = np.sqrt(np.sum(np.square(accs[:] - means)) / cross_validations)
print("Current Mean: " + str(means) + " current STD: " + str(stds))

# Mean:
mean = np.mean(accs)

# standard deviation:
std = np.sqrt(np.sum(np.square(accs - mean)) / cross_validations)

print("Accuracies: " + str(accs))
print("Means: " + str(means))
print("Standard Deviations: " + str(stds))

# Storing accs, mean, std as JSON

d = {"conf_mats": conf_mats.tolist(), "conf_mat_mean": mean_cvs.tolist(), "conf_mats_std": std_cvs.tolist(),
     "accuracies": accs.tolist(), "acc_mean": means.tolist(), "acc_std": stds.tolist(), "learning_rates": learning}


'''
current_destination_file = 'metrics_siamese_20cv20ep_' + files1024[0] + '.json'
# with open("metrics_modpcen_lkwpkw_N512fs16kHz_balanced.json", "w") as f:
with open(current_destination_file, "w") as f:
    json.dump(d, f)
'''