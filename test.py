# from kan import *
# import torch
# import torch.nn as nn

# from datasets import load_dataset



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #load the MNIST dataset and reduce the size of the images to 7x7

# dataset = load_dataset("mnist")
# X_train, y_train = dataset["train"]["image"][:1000], dataset["train"]["label"][:1000]
# X_test, y_test = dataset["test"]["image"][:100], dataset["test"]["label"][:100]

# #downsample the images with torch
# downsample = nn.AvgPool2d(4)
# X_train = downsample(torch.tensor(X_train, dtype=torch.float32)).numpy()
# X_test = downsample(torch.tensor(X_test, dtype=torch.float32)).numpy()


# #flatten the images

# X_train = X_train.reshape(X_train.shape[0], -1)
# X_test = X_test.reshape(X_test.shape[0], -1)

# #normalize the images
# X_train = X_train / 255
# X_test = X_test / 255

# #create a dataset : dic
#                 #contains dataset['train_input'], dataset['train_label'], dataset['test_input'], dataset['test_label']
# dataset = {
#     "train_input": torch.tensor(X_train, dtype=torch.float32).to(device),
#     "train_label": torch.tensor(y_train, dtype=torch.long).to(device),
#     "test_input": torch.tensor(X_test, dtype=torch.float32).to(device),
#     "test_label": torch.tensor(y_test, dtype=torch.long).to(device)
# }


# #initialize model
# model = KAN(width=[7*7, 5, 5, 128], grid=3, k=3, device=device)




# model.train(dataset, opt="LBFGS", steps=20, batch=128)

import torch
n_labels = 4

# Creare una matrice NxM
matrix = torch.randn(32,128,16) #Batch size, sequence length, embedding dimension



# Creare un vettore Mx1
vector = torch.randn(16,n_labels);

# Moltiplicare la matrice per il vettore
result = torch.matmul(matrix, vector)

print(result)  # Output: tensor([ 4., 10., 16.]
print(result.shape)  # Output: torch.Size([3]
