import torch

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


from lib import modelv0
from lib import modelv7
from lib import modelv8
from lib import modelv9


from lib import utils

from lib.dataset import dataset_importer
from lib import utils

VOCAB_SIZE = 512*8
BATCH_SIZE = 32

REDUCED_EMBEDDING_DIM = 16
EMBED_DIM = 128
NUM_HEADS = 8
FORWARD_EXPANSION = 0.1
MAX_LENGTH = 128
LAYERS = 1




########################################################################################
print("Loading dataset:")

#'Amazon', "imdb", "sst2" "sst5" "twitter" "race" "yelp"
DATASET = "race"

train_dataloader, test_dataloader, N_LABELS, label_test = dataset_importer(DATASET, VOCAB_SIZE, MAX_LENGTH, BATCH_SIZE)


########################################################################################
print("Model initialization:")
#initialize model
# classifier = modelv0.ClassifierV0(VOCAB_SIZE, MAX_LENGTH, EMBED_DIM, NUM_HEADS, FORWARD_EXPANSION, N_LABELS)
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
LR = 2e-4

utils.trainer(classifier, train_dataloader, LR, EPOCHS)

########################################################################################
print("Starting evaluation")
predicted = utils.evaluator(classifier, test_dataloader)

print(classification_report(label_test, predicted))

conf_matrix = confusion_matrix(label_test, predicted)

print("Confusion matrix:\n", conf_matrix)


########################################################################################
#save model
# t_string = "./models/transformer_v7_" + str(REDUCED_EMBEDDING_DIM) + "_" + str(EMBED_DIM) + "_" + str(FORWARD_EXPANSION) + "_" + str(LAYERS) + "_" + str(VOCAB_SIZE) + "_" + str(MAX_LENGTH) + "_" + str(BATCH_SIZE) + "_" + str(EPOCHS) + "_" + str(DATASET) + ".pt"
# torch.save(classifier.state_dict(), t_string)