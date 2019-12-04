import src.siamese_nn_raw as nn
import json
import numpy as np


epochs = 15
learning = 0.00008
b = 1
f = [16, 8]
k = [[5,5], [3,3]]
p = [[2,2],[2,2]]
d = 0.8
u = 256


# The Features in the same order as in Becker et al. (Fig. 2, Non-Modulation Features)
files = ['features/pcen_dp.mat',
         'features/mfccs_dp.mat',
         'features/pcen_hgvpc.mat',
         'features/mfccs_hgvpc.mat']

# Index for Feature Set selection
file_index = 0

# Preallocation of the Confusion Matrix that will be later used as a requisite for the F1-Score
cross_validations = 20
conf_mats = np.zeros((cross_validations, 2, 2))
mean_cvs = np.zeros((2, 2))
std_cvs = np.zeros((2, 2))

# Optional Metric: Accuracy
accs = np.zeros(cross_validations)


# A loop that computes the Confusion Matrix over all Cross Validations for a given feature set
for i in range(cross_validations):
    print("Cross-Validation " + str(i + 1) + " of " + str(cross_validations))
    accs[i], conf_mats[i, :, :] = nn.train_nn(files[file_index], learning, epochs, [0.15, 0.2], b, f, k, p, d, u)


# Statistical Properties of the Confusion Matrix (Mean, Standard Deviation)
# Mean
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

# Mean (Accuracy):
mean = np.mean(accs)

# Standard Deviation (Accuracy):
std = np.sqrt(np.sum(np.square(accs - mean)) / cross_validations)

print("Accuracies: " + str(accs))
print("Means: " + str(means))
print("Standard Deviations: " + str(stds))

# Storing Accuracies, Confusion Matrices and their respective statistical properties as a JSON

d = {"conf_mats": conf_mats.tolist(), "conf_mat_mean": mean_cvs.tolist(), "conf_mats_std": std_cvs.tolist(),
     "accuracies": accs.tolist(), "acc_mean": means.tolist(), "acc_std": stds.tolist(), "learning_rates": learning}

current_destination_file = 'metrics_' + files[file_index] + '.json'
with open(current_destination_file, "w") as f:
    json.dump(d, f)
