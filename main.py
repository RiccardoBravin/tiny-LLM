import torch

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


from models import modelv7
from models import utils

from models.dataset import dataset_importer
from models import utils

VOCAB_SIZE = 512*8
BATCH_SIZE = 32

REDUCED_EMBEDDING_DIM = 16
EMBED_DIM = 124
NUM_HEADS = 8
FORWARD_EXPANSION = 0.1
MAX_LENGTH = 512
LAYERS = 1


print("Loading dataset:")

#'Amazon', "imdb", "sst2" "sst5" "twitter"
DATASET = "imdb"

train_dataloader, test_dataloader, N_LABELS, label_test = dataset_importer(DATASET, VOCAB_SIZE, MAX_LENGTH, BATCH_SIZE)

print("Model initialization:")
#initialize model
classifier = modelv7.Classifier(VOCAB_SIZE, MAX_LENGTH, REDUCED_EMBEDDING_DIM, EMBED_DIM, NUM_HEADS, FORWARD_EXPANSION, LAYERS, N_LABELS)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier.to(device)

print("Model parameters:")
#print all model parameters with names
for name, param in classifier.named_parameters():
  print(f"{name}: {param.nelement()}")

#print the model size
utils.print_model_size(classifier)




print("Starting training")
EPOCHS = 10
LR = 2e-4


utils.trainer(classifier, train_dataloader, LR, EPOCHS)

predicted = utils.evaluator(classifier, test_dataloader)

print(classification_report(label_test, predicted))

conf_matrix = confusion_matrix(label_test, predicted)

print("Confusion matrix:\n", conf_matrix)

# Print or visualize the confusion matrix
# print("Confusion Matrix:")
# print(conf_matrix)