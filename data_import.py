import scipy.io as scio

data = scio.loadmat("./symbols_dataset_final.mat")

label_matrix = data["LableMatrix"]
print(len(label_matrix[0]))
