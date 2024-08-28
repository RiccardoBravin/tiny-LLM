#File to plot the two losses and the accuracy of the model

#Example of the output of the file to read with data
# First launch of NanoBERT efficient

# Step 63029/2312633:              MLM Loss: 3.680 CLS Loss: 0.502 CLS Accuracy: 0.723 CLS MCC: 0.450                                                                                     
# Step 63252/2312633:              MLM Loss: 3.685 CLS Loss: 0.550 CLS Accuracy: 0.759 CLS MCC: 0.533                                                                                     
# Step 63473/2312633:              MLM Loss: 3.713 CLS Loss: 0.597 CLS Accuracy: 0.728 CLS MCC: 0.467                                                                                     
# Step 63695/2312633:              MLM Loss: 4.363 CLS Loss: 0.564 CLS Accuracy: 0.707 CLS MCC: 0.419                                                                                     
# Step 63917/2312633:              MLM Loss: 3.968 CLS Loss: 0.682 CLS Accuracy: 0.692 CLS MCC: 0.395                                                                                     
# Step 64138/2312633:              MLM Loss: 3.320 CLS Loss: 0.654 CLS Accuracy: 0.653 CLS MCC: 0.336                                                                                     
# Step 64360/2312633:              MLM Loss: 4.063 CLS Loss: 0.514 CLS Accuracy: 0.673 CLS MCC: 0.388                                                                                     
# Step 64582/2312633:              MLM Loss: 3.051 CLS Loss: 0.509 CLS Accuracy: 0.686 CLS MCC: 0.405                                                                                     
# Step 64803/2312633:              MLM Loss: 3.187 CLS Loss: 0.644 CLS Accuracy: 0.649 CLS MCC: 0.325                                                                                     
 

import matplotlib.pyplot as plt
import numpy as np
import torch

#Read the file
FILE_NAME = "trained_models/logs/log_nanobert.txt"
with open(FILE_NAME, 'r') as f:
    lines = f.readlines()

#Extract the data
steps = []
mlm_losses = []
cls_losses = []
cls_accuracies = []
cls_mccs = []

for line in lines:
    if 'Step' in line:
        step = int(line.split('/')[0].split(' ')[1])
        steps.append(step)
        mlm_loss = float(line.split('MLM Loss: ')[1].split(' ')[0])
        mlm_losses.append(mlm_loss)
        cls_loss = float(line.split('CLS Loss: ')[1].split(' ')[0])
        cls_losses.append(cls_loss)
        cls_accuracy = float(line.split('CLS Accuracy: ')[1].split(' ')[0])
        cls_accuracies.append(cls_accuracy)
        cls_mcc = float(line.split('CLS MCC: ')[1].split(' ')[0])
        cls_mccs.append(cls_mcc)

#Plot the data in a graph
#The actual data is plotted semitransparently, and the smoothed data is plotted in full color
#The smoothed data is computed by averaging the data in a window of size WINDOW_SIZE
WINDOW_SIZE = 10

fig, ax = plt.subplots(3, 1, figsize=(10, 10))

ax[0].plot(range(len(mlm_losses)), mlm_losses, alpha=0.3, label='MLM Loss')
ax[0].plot(np.convolve(mlm_losses, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='valid'), label='Smoothed MLM Loss')
ax[0].set_xlabel('Step')
ax[0].set_ylabel('Loss')
ax[0].legend()

ax[1].plot(range(len(cls_losses)), cls_losses, alpha=0.3, label='CLS Loss')
ax[1].plot(np.convolve(cls_losses, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='valid'), label='Smoothed CLS Loss')
ax[1].set_xlabel('Step')
ax[1].set_ylabel('Loss')
ax[1].legend()

ax[2].plot(range(len(cls_accuracies)), cls_accuracies, alpha=0.3, label='CLS Accuracy')
ax[2].plot(np.convolve(cls_accuracies, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='valid'), label='Smoothed CLS Accuracy')
ax[2].set_xlabel('Step')
ax[2].set_ylabel('Accuracy')
ax[2].legend()



fig.tight_layout()
plt.show()

