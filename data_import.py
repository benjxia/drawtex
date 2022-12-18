import scipy.io as scio

vars = scio.loadmat("./symbols_dataset_final.mat")

label_matrix = vars["LableMatrix"]
print(len(label_matrix[0]))
