import torch

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


from lib import modelv0
from lib import modelv7
from lib import modelv8
from lib import modelv9
from lib import modelv10
from lib import MLPLLM


from lib import utils

from lib.dataset import dataset_importer
from lib import utils

# import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

VOCAB_SIZE = 512*8
BATCH_SIZE = 32

REDUCED_EMBEDDING_DIM = 16
EMBED_DIM = 128
NUM_HEADS = 8
FORWARD_EXPANSION = 0.5
MAX_LENGTH = 64
LAYERS = 1




########################################################################################
print("Loading dataset:")

#'Amazon', "imdb", "sst2" "sst5" "twitter" "race" "yelp" "news" "trec_coarse" "bull"
DATASET = "sst2"

train_dataloader, test_dataloader, N_LABELS, label_test = dataset_importer(DATASET, VOCAB_SIZE, MAX_LENGTH, BATCH_SIZE)


########################################################################################
print("Model initialization:")
#initialize model
#classifier = modelv0.ClassifierV0(VOCAB_SIZE, MAX_LENGTH, EMBED_DIM, NUM_HEADS, FORWARD_EXPANSION, N_LABELS)
# classifier = modelv7.ClassifierV7(VOCAB_SIZE, MAX_LENGTH, REDUCED_EMBEDDING_DIM, EMBED_DIM, NUM_HEADS, FORWARD_EXPANSION, LAYERS, N_LABELS)
classifier = modelv9.ClassifierV9(VOCAB_SIZE, MAX_LENGTH, REDUCED_EMBEDDING_DIM, EMBED_DIM, NUM_HEADS, FORWARD_EXPANSION, LAYERS, N_LABELS)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier.to(device)
print(f"Model initialized on {device}")


print("Model parameters:")
#print all model parameters with names
for name, param in classifier.named_parameters():
	print(f"{name}: {param.nelement()}")

#print the model size
utils.print_model_size(classifier)



########################################################################################
print("Starting training")

EPOCHS = 20
LR = 1e-3

utils.trainer(classifier, train_dataloader, LR, EPOCHS)

########################################################################################
print("Starting evaluation")
predicted, avg_loss = utils.evaluator(classifier, test_dataloader)

print(classification_report(label_test, predicted))

conf_matrix = confusion_matrix(label_test, predicted)

print("Confusion matrix:\n", conf_matrix)


########################################################################################
#save the classification report in a file for later use specifying the dataset, model hyperparameters
with open(f"results/{DATASET}_classification_report.txt", "a") as f:
	f.write(f"MODEL: {classifier.__class__.__name__}\n")
	f.write(f"DATASET: {DATASET}\n")
	f.write(f"VOCAB_SIZE: {VOCAB_SIZE}\n")
	f.write(f"EMBED_DIM: {EMBED_DIM}\n")
	f.write(f"FORWARD_EXPANSION: {FORWARD_EXPANSION}\n")
	f.write(f"MAX_LENGTH: {MAX_LENGTH}\n")
	f.write(f"LAYERS: {LAYERS}\n")
	f.write(f"REDUCED_EMBEDDING_DIM: {REDUCED_EMBEDDING_DIM}\n")
	f.write(f"LR: {LR}\n")
	f.write(f"EPOCHS: {EPOCHS}\n\n")
	f.write(f"Avg loss: {avg_loss}\n")
	f.write(classification_report(label_test, predicted))
	f.write("\n")
	f.write("Confusion matrix:\n")
	f.write(str(conf_matrix))
	f.write("\n\n*******************************************\n\n")
