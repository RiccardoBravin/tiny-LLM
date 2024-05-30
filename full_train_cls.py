#COLORS
from colors import ATTRIBUTES, FOREGROUND_COLORS, RESET

#STD
import os
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix


#CUSTOM
from lib import utils
from lib.dataset import dataset_importer
from lib.MLP import MLPSwiGLU
from lib.BERT_Eff import BERT_efficient
from lib.BERT_Eff_Enc_Head import BERT_Eff_Multihead
from lib.GatedBert import Gated_BERT
from lib.BRAV_multihead import BRAV_multihead
from lib.BRAV_2 import BRAV_2

from lib.classifier import classifier

torch.set_printoptions(profile="full")

VOCAB_SIZE = 512*8
BATCH_SIZE = 128

REDUCED_EMBEDDING_DIM = 16
EMBED_DIM = 128
NUM_HEADS = 8
FORWARD_EXPANSION = 2
MAX_LENGTH = 512
LAYERS = 4

########################################################################################

#'Amazon', "imdb", "sst2" "twitter" "race" "yelp" "news" "trec_coarse" "bull" "limit" "dbpedia" "nlu" "snips" "blog" "multi_nli"
for DATASET in ["imdb", "sst2", "news", "bull", "limit", "dbpedia", "nlu", "snips", "multi_nli"]:
	print(f"{ATTRIBUTES['Bold']}Loading dataset {DATASET}: {RESET}")


	train_dataloader, val_dataloader, test_dataloader, LABELS, label_test = dataset_importer(DATASET, VOCAB_SIZE, MAX_LENGTH, BATCH_SIZE)		
	N_LABELS = len(LABELS)
	########################################################################################
	for _ in range(5):
		print(f"{ATTRIBUTES['Bold']}Models initialization:{RESET}")

		models = [
			#MLPSwiGLU(VOCAB_SIZE, EMBED_DIM, LAYERS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1),
			#BERT_efficient(VOCAB_SIZE, EMBED_DIM, LAYERS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1),
			BERT_Eff_Multihead(VOCAB_SIZE, EMBED_DIM, REDUCED_EMBEDDING_DIM, LAYERS, NUM_HEADS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1),
			#BRAV_multihead(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, LAYERS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1),
			BRAV_2(VOCAB_SIZE, EMBED_DIM//4, REDUCED_EMBEDDING_DIM ,LAYERS+1, MAX_LENGTH, FORWARD_EXPANSION*4, dropout=0.1),
			#Gated_BERT(VOCAB_SIZE, EMBED_DIM, LAYERS, NUM_HEADS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)

		]

		for model in models:
			cls = classifier(model, EMBED_DIM, N_LABELS)

			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			cls.to(device)
			print(f"Model initialized on {device}")

			print("Model parameters:")
			#print all model parameters with names
			for name, param in cls.named_parameters():
				print(f"{name}: {param.nelement()}")

			print(utils.model_size(cls))


			########################################################################################
			print(f"{ATTRIBUTES['Bold']}Starting training of model {model.__class__.__name__}{RESET}")

			EPOCHS = 10
			LR = 1e-2


			utils.trainer(cls, train_dataloader, val_dataloader, LR, EPOCHS)

			########################################################################################
			print(f"{ATTRIBUTES['Bold']}Starting evaluation{RESET}")
			predicted, avg_eval_loss = utils.evaluator(cls, test_dataloader)

			########################################################################################
			accuracy = accuracy_score(label_test, predicted)
			f1 = f1_score(label_test, predicted, average='weighted')  
			precision = precision_score(label_test, predicted, average='weighted')  
			recall = recall_score(label_test, predicted, average='weighted')  
			mcc = matthews_corrcoef(label_test, predicted)
			conf_mat = confusion_matrix(label_test, predicted)

			print(f"{FOREGROUND_COLORS['Red']}")
			print(f"Accuracy: {accuracy}")
			print(f"F1: {f1}")
			print(f"Precision: {precision}")
			print(f"Recall: {recall}")
			print(f"MCC: {mcc}")
			print(f"Confusion matrix:\n {conf_mat}")
			print(f"{RESET}")
			########################################################################################
			if not os.path.exists(f"results/{model.__class__.__name__}/"):
				os.makedirs(f"results/{model.__class__.__name__}/")

			#save the classification report in a file for later use specifying the dataset, model hyperparameters
			with open(f"results/{model.__class__.__name__}/cls_{DATASET}_classification_report.txt", "a") as f:
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
				f.write(str(utils.model_size(cls)))
				f.write("\n\n*******************************************\n\n")

	with open(f"results/{model.__class__.__name__}/cls_{DATASET}_classification_report.txt", "a") as f:
		f.write(f"**************************************************\n==================================================\n**************************************************\n")
