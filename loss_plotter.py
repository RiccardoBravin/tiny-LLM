#File to plot the two losses and the accuracy of the model
                                                                                 

import matplotlib.pyplot as plt
import numpy as np
import torch

#Read the file
FILE_NAME = "trained_models/logs/Nano_Bert_Efficient_mh_train_log.txt"
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

#make title the name of the file after last '/' and before '_train_log.txt'
plt.suptitle(FILE_NAME.split('/')[-1].split('_train_log.txt')[0])

fig.tight_layout()
plt.show()

