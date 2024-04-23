import torch
from torch.autograd import Variable
from torch import nn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from lib.BERT import BERT
from lib.dataset import dataset_importer
from lib.utils import trainer, evaluator

# Print the model size
def print_model_size(model):
	param_size = 0
	param_count = 0
	for param in model.parameters():
		param_size += param.nelement() * param.element_size()
		param_count += param.nelement()
	buffer_size = 0
	for buffer in model.buffers():
		buffer_size += buffer.nelement() * buffer.element_size()

	size_all_mb = (param_size + buffer_size) / 1024**2
	print('Model params: {:.3f}M'.format(param_count/1e6))
	print('Model size: {:.3f}MB'.format(size_all_mb))

# Define parameters
vocab_size = 512*8 # Example vocabulary size
maxlen = 512  # Maximum sequence length
d_model = 128  # Dimension of token embeddings
n_layers = 1  # Number of encoder layers
ff_exp = 4  # Forward expansion factor
heads = 8  # Number of attention head

BATCH_SIZE = 32  # Example batch size


########################################################################################
print("Loading dataset:")

#'Amazon', "imdb", "sst2" "sst5" "twitter" "race" "yelp"
DATASET = "race"

train_dataloader, test_dataloader, N_LABELS, label_test = dataset_importer(DATASET, vocab_size, maxlen, BATCH_SIZE)

# Create model instance
model = BERT(vocab_size, d_model, n_layers, maxlen, ff_exp, heads, N_LABELS)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Model initialized on {device}")

print("Model parameters:")
#print all model parameters with names
for name, param in model.named_parameters():
	print(f"{name}: {param.nelement()}")
print_model_size(model)


trainer(model, train_dataloader, 2e-4, 5)

########################################################################################
print("Starting evaluation")
predicted = evaluator(model, test_dataloader)

print(classification_report(label_test, predicted))

conf_matrix = confusion_matrix(label_test, predicted)

print("Confusion matrix:\n", conf_matrix)
