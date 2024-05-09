import torch
import evaluate

from colors import ATTRIBUTES, FOREGROUND_COLORS, RESET

from lib import utils
from lib.MAMBA import Mamba, MambaConfig, Mamba_classifier
from lib.BERT_Eff import BERT_efficient
from lib.BERT_Eff_Enc_Gray import BERT_Eff_gray
from lib.BERT_Eff_Enc_Head import BERT_Eff_Multihead
from lib.BRAV import BRAV
from lib.MLP import MLPSwiGLU
from lib.GatedBert import Gated_BERT

from lib.classifier import classifier

from lib.dataset import dataset_importer

VOCAB_SIZE = 512*8
BATCH_SIZE = 512

REDUCED_EMBEDDING_DIM = 16
EMBED_DIM = 128
NUM_HEADS = 8
FORWARD_EXPANSION = 2
MAX_LENGTH = 64
LAYERS = 6

########################################################################################
print(f"{ATTRIBUTES['Bold']}Loading dataset:{RESET}")

#'Amazon', "imdb", "sst2" "sst5" "twitter" "race" "yelp" "news" "trec_coarse" "bull"
DATASET = "bull"

train_dataloader, val_dataloader, test_dataloader, N_LABELS, label_test = dataset_importer(DATASET, VOCAB_SIZE, MAX_LENGTH, BATCH_SIZE)


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
model = Gated_BERT(VOCAB_SIZE, EMBED_DIM, LAYERS, NUM_HEADS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)


classifier = classifier(model, EMBED_DIM, N_LABELS)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier.to(device)
print(f"Model initialized on {device}")


print("Model parameters:")
#print all model parameters with names
for name, param in classifier.named_parameters():
	print(f"{name}: {param.nelement()}")

#print the model size
print(utils.model_size(classifier))



########################################################################################
print(f"{ATTRIBUTES['Bold']}Starting training{RESET}")

EPOCHS = 5
LR = 1e-2

utils.trainer(classifier, train_dataloader, val_dataloader, LR, EPOCHS)

########################################################################################
print(f"{ATTRIBUTES['Bold']}Starting evaluation{RESET}")
predicted, avg_eval_loss = utils.evaluator(classifier, test_dataloader)

accuracy = evaluate.load("accuracy").compute(references=label_test, predictions=predicted)
f1 = evaluate.load("f1").compute(references=label_test, predictions=predicted, average="weighted")
precision = evaluate.load("precision").compute(references=label_test, predictions=predicted, average="weighted", zero_division=0)
recall = evaluate.load("recall").compute(references=label_test, predictions=predicted, average="weighted")
mcc = evaluate.load("matthews_correlation").compute(references=label_test, predictions=predicted, average="weighted")
conf_mat = evaluate.load("confusion_matrix").compute(references=label_test, predictions=predicted)

print(f"Accuracy: {accuracy['accuracy']}")
print(f"F1: {f1['f1']}")
print(f"Precision: {precision['precision']}")
print(f"Recall: {recall['recall']}")
print(f"MCC: {mcc['matthews_correlation']}")
print(f"Confusion matrix:\n {conf_mat['confusion_matrix']}")

########################################################################################
#save the classification report in a file for later use specifying the dataset, model hyperparameters
# with open(f"results/{DATASET}_classification_report.txt", "a") as f:
# 	f.write(f"MODEL: {classifier.__class__.__name__}\n")
# 	f.write(f"DATASET: {DATASET}\n")
# 	f.write(f"VOCAB_SIZE: {VOCAB_SIZE}\n")
# 	f.write(f"EMBED_DIM: {EMBED_DIM}\n")
# 	f.write(f"FORWARD_EXPANSION: {FORWARD_EXPANSION}\n")
# 	f.write(f"MAX_LENGTH: {MAX_LENGTH}\n")
# 	f.write(f"LAYERS: {LAYERS}\n")
# 	f.write(f"REDUCED_EMBEDDING_DIM: {REDUCED_EMBEDDING_DIM}\n")
# 	f.write(f"LR: {LR}\n")
# 	f.write(f"EPOCHS: {EPOCHS}\n\n")
# 	f.write(f"average eval loss: {avg_eval_loss}\n")
# 	f.write(f"Accuracy: {accuracy['accuracy']}\n")
# 	f.write(f"F1: {f1['f1']}\n")
# 	f.write(f"Precision: {precision['precision']}\n")
# 	f.write(f"Recall: {recall['recall']}\n")
# 	f.write(f"MCC: {mcc['matthews_correlation']}\n")
# 	f.write(f"Confusion matrix:\n {conf_mat['confusion_matrix']}\n")
# 	f.write(str(utils.model_size(classifier)))
# 	f.write("\n\n*******************************************\n\n")
