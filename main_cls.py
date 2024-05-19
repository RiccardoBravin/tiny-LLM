import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix

from colors import ATTRIBUTES, FOREGROUND_COLORS, RESET

from lib import utils
from lib.MAMBA import Mamba, MambaConfig, Mamba_classifier
from lib.BERT_Eff import BERT_efficient
from lib.BERT_Eff_Enc_Gray import BERT_Eff_gray
from lib.BERT_Eff_Enc_Head import BERT_Eff_Multihead
from lib.BRAV import BRAV
from lib.BRAV_2 import BRAV_2
from lib.BRAV_multihead import BRAV_multihead
from lib.MLP import MLPSwiGLU
from lib.GatedBert import Gated_BERT

from lib.classifier import classifier

from lib.dataset import dataset_importer

VOCAB_SIZE = 512*8
BATCH_SIZE = 128

REDUCED_EMBEDDING_DIM = 16
#EMBED_DIM = 128
EMBED_DIM = 32
NUM_HEADS = 8
FORWARD_EXPANSION = 4
MAX_LENGTH = 64
LAYERS = 4

########################################################################################
print(f"{ATTRIBUTES['Bold']}Loading dataset:{RESET}")

#'Amazon', "imdb", "sst2" "twitter" "race" "yelp" "news" "trec_coarse" "bull" "limit" "dbpedia" "nlu" "snips" "blog"
DATASET = "bull"

train_dataloader, val_dataloader, test_dataloader, LABELS, label_test = dataset_importer(DATASET, VOCAB_SIZE, MAX_LENGTH, BATCH_SIZE)
N_LABELS = len(LABELS)

########################################################################################
print(f"{ATTRIBUTES['Bold']}Model initialization:{RESET}")

# config = MambaConfig(d_model=EMBED_DIM, n_layers=LAYERS)
# model = Mamba(config)
# classifier = Mamba_classifier(model, EMBED_DIM, REDUCED_EMBEDDING_DIM, VOCAB_SIZE, N_LABELS)

# model = MLPSwiGLU(VOCAB_SIZE, EMBED_DIM, LAYERS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)
# model = BERT_Efficient(VOCAB_SIZE, EMBED_DIM, LAYERS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)
# model = BERT_Eff_gray(VOCAB_SIZE, EMBED_DIM, REDUCED_EMBEDDING_DIM, LAYERS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)
# model = BERT_Eff_multihead(VOCAB_SIZE, EMBED_DIM, LAYERS, NUM_HEADS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)	
# model = BRAV(VOCAB_SIZE, EMBED_DIM, LAYERS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)
#model = BRAV_multihead(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, LAYERS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)
model = BRAV_2(VOCAB_SIZE, EMBED_DIM, LAYERS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)
#model = Gated_BERT(VOCAB_SIZE, EMBED_DIM, LAYERS, NUM_HEADS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)


cls = classifier(model, EMBED_DIM, N_LABELS)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cls.to(device)
print(f"Model initialized on {device}")


print("Model parameters:")
#print all model parameters with names
for name, param in cls.named_parameters():
	print(f"{name}: {param.nelement()}")

#print the model size
print(utils.model_size(cls))



########################################################################################
print(f"{ATTRIBUTES['Bold']}Starting training{RESET}")

EPOCHS = 10
LR = 1e-2

utils.trainer(cls, train_dataloader, val_dataloader, LR, EPOCHS)

########################################################################################
print(f"{ATTRIBUTES['Bold']}Starting evaluation{RESET}")
predicted, avg_eval_loss = utils.evaluator(cls, test_dataloader)

accuracy = accuracy_score(label_test, predicted)
f1 = f1_score(label_test, predicted, average='micro')  
precision = precision_score(label_test, predicted, average='micro')  
recall = recall_score(label_test, predicted, average='micro')
mcc = matthews_corrcoef(label_test, predicted)
conf_mat = confusion_matrix(label_test, predicted)

print(f"Accuracy: {accuracy}")
print(f"F1: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"MCC: {mcc}")
print(f"Confusion matrix:\n {conf_mat}")

