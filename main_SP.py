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
from lib.MLP import MLPSwiGLU
from lib.GatedBert import Gated_BERT

from lib.classifier import classifier
from lib.classifier import classifier_SP
from lib.Electra import SimpleElectra, electraTrainer, electraEvaluator

from lib.dataset import dataset_importer

VOCAB_SIZE = 512*8
BATCH_SIZE = 128

REDUCED_EMBEDDING_DIM = 16
EMBED_DIM = 32
NUM_HEADS = 8
FORWARD_EXPANSION = 4
MAX_LENGTH = 128
LAYERS = 4

########################################################################################
print(f"{ATTRIBUTES['Bold']}Loading dataset:{RESET}")

#'Amazon', "imdb", "sst2" "twitter" "race" "yelp" "news" "trec_coarse" "bull" "limit" "dbpedia" "nlu" "snips" "limit" "blog"
DATASET = "bull"

train_dataloader, val_dataloader, test_dataloader, LABELS, label_test = dataset_importer(DATASET, VOCAB_SIZE, MAX_LENGTH, BATCH_SIZE)
N_LABELS = len(LABELS)

########################################################################################
print(f"{ATTRIBUTES['Bold']}Electra model initialization:{RESET}")

# config = MambaConfig(d_model=EMBED_DIM, n_layers=LAYERS)
# model = Mamba(config)
# classifier = Mamba_classifier(model, EMBED_DIM, REDUCED_EMBEDDING_DIM, VOCAB_SIZE, N_LABELS)

# model = MLPSwiGLU(VOCAB_SIZE, EMBED_DIM, LAYERS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)
# model = BERT_efficient(VOCAB_SIZE, EMBED_DIM, LAYERS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)
# model = BERT_Eff_gray(VOCAB_SIZE, EMBED_DIM, REDUCED_EMBEDDING_DIM, LAYERS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)
# model = BERT_Eff_Multihead(VOCAB_SIZE, EMBED_DIM, LAYERS, NUM_HEADS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)	
# model = BRAV(VOCAB_SIZE, EMBED_DIM, LAYERS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)
#model = Gated_BERT(VOCAB_SIZE, EMBED_DIM, LAYERS, NUM_HEADS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)
model = BRAV_2(VOCAB_SIZE, EMBED_DIM, LAYERS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)

electra_cls = SimpleElectra(model, EMBED_DIM, VOCAB_SIZE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
electra_cls.to(device)
print(f"Model initialized on {device}")


print("Model parameters:")
#print all model parameters with names
for name, param in electra_cls.named_parameters():
	print(f"{name}: {param.nelement()}")

#print the model size
print(utils.model_size(electra_cls))



########################################################################################
print(f"{ATTRIBUTES['Bold']}Starting training{RESET}")

EPOCHS = 20
LR = 1e-2

electraTrainer(electra_cls, train_dataloader, val_dataloader, LR, EPOCHS)

########################################################################################
print(f"{ATTRIBUTES['Bold']}Starting evaluation{RESET}")
predicted, avg_eval_loss, created_labels = electraEvaluator(electra_cls, test_dataloader)


accuracy = accuracy_score(created_labels.cpu(), predicted)
f1 = f1_score(created_labels.cpu(), predicted, average='micro')  
precision = precision_score(created_labels.cpu(), predicted, average='micro')  
recall = recall_score(created_labels.cpu(), predicted, average='micro')
mcc = matthews_corrcoef(created_labels.cpu(), predicted)
conf_mat = confusion_matrix(created_labels.cpu(), predicted)


print(f"Accuracy: {accuracy}")
print(f"F1: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"MCC: {mcc}")
print(f"Confusion matrix:\n {conf_mat}")


with open(f"results/tests_{DATASET}_classification_report.txt", "a") as f:
	f.write(f"Values of ELECTRA training\n")
	f.write(f"MODEL: {model.__class__.__name__}\n")
	f.write(f"DATASET: {DATASET}\n")
	f.write(f"VOCAB_SIZE: {VOCAB_SIZE}\n")
	f.write(f"EMBED_DIM: {EMBED_DIM}\n")
	f.write(f"FORWARD_EXPANSION: {FORWARD_EXPANSION}\n")
	f.write(f"MAX_LENGTH: {MAX_LENGTH}\n")
	f.write(f"LAYERS: {LAYERS}\n")
	f.write(f"REDUCED_EMBEDDING_DIM: {REDUCED_EMBEDDING_DIM}\n")
	f.write(f"LR: {LR}\n")
	f.write(f"EPOCHS: {EPOCHS}\n\n")
	f.write(f"average eval loss: {avg_eval_loss}\n")
	f.write(f"average eval loss: {avg_eval_loss}\n")
	f.write(f"Accuracy: {accuracy}\n")
	f.write(f"F1 (weighted): {f1}\n")
	f.write(f"Precision (weighted): {precision}\n")
	f.write(f"Recall (weighted): {recall}\n")
	f.write(f"MCC: {mcc}\n")
	f.write(f"Confusion matrix:\n {conf_mat}\n")
	f.write(str(utils.model_size(electra_cls)))
	f.write("\n\n*******************************************\n\n")












########################################################################################
print(f"{ATTRIBUTES['Bold']}Normal model initialization:{RESET}")

std_cls = classifier_SP(model, EMBED_DIM, MAX_LENGTH, N_LABELS)
std_cls.to(device)
print(f"Model initialized on {device}")


print("Model parameters:")
#print all model parameters with names
for name, param in std_cls.named_parameters():
	print(f"{name}: {param.nelement()}")

#print the model size
print(utils.model_size(std_cls))

########################################################################################

print(f"{ATTRIBUTES['Bold']}Starting training{RESET}")

EPOCHS = 10
LR = 1e-3

utils.trainer(std_cls, train_dataloader, val_dataloader, LR, EPOCHS)


########################################################################################
print(f"{ATTRIBUTES['Bold']}Starting evaluation{RESET}")
predicted, avg_eval_loss = utils.evaluator(std_cls, test_dataloader)

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



########################################################################################
#save the classification report in a file for later use specifying the dataset, model hyperparameters
with open(f"results/tests_{DATASET}_classification_report.txt", "a") as f:
	f.write(f"Values of final training\n")
	f.write(f"MODEL: {model.__class__.__name__}\n")
	f.write(f"DATASET: {DATASET}\n")
	f.write(f"VOCAB_SIZE: {VOCAB_SIZE}\n")
	f.write(f"EMBED_DIM: {EMBED_DIM}\n")
	f.write(f"FORWARD_EXPANSION: {FORWARD_EXPANSION}\n")
	f.write(f"MAX_LENGTH: {MAX_LENGTH}\n")
	f.write(f"LAYERS: {LAYERS}\n")
	f.write(f"REDUCED_EMBEDDING_DIM: {REDUCED_EMBEDDING_DIM}\n")
	f.write(f"LR: {LR}\n")
	f.write(f"EPOCHS: {EPOCHS}\n\n")
	f.write(f"average eval loss: {avg_eval_loss}\n")
	f.write(f"Accuracy: {accuracy}\n")
	f.write(f"F1 (weighted): {f1}\n")
	f.write(f"Precision (weighted): {precision}\n")
	f.write(f"Recall (weighted): {recall}\n")
	f.write(f"MCC: {mcc}\n")
	f.write(f"Confusion matrix:\n {conf_mat}\n")
	f.write(str(utils.model_size(std_cls)))
	f.write("\n\n*******************************************\n\n")



