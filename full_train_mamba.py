#COLORS
from colors import ATTRIBUTES, FOREGROUND_COLORS, RESET

#STD
import os
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix


#CUSTOM
from lib import utils
from lib.dataset import dataset_importer

from lib.MAMBA import Mamba, MambaConfig, Mamba_classifier

from lib.classifier import classifier

torch.set_printoptions(profile="full")

VOCAB_SIZE = 512*8
BATCH_SIZE = 128

REDUCED_EMBEDDING_DIM = 16
EMBED_DIM = 128
NUM_HEADS = 8
FORWARD_EXPANSION = 2
MAX_LENGTH = 512
LAYERS = 1

########################################################################################

#'Amazon', "imdb", "sst2" "twitter" "race" "yelp" "news" "trec_coarse" "bull" "limit" "dbpedia" "nlu" "snips" "blog" "multi_nli"
for DATASET in ["imdb", "sst2", "news", "bull", "limit", "dbpedia", "nlu", "snips", "blog", "multi_nli"]:
	print(f"{ATTRIBUTES['Bold']}Loading dataset {DATASET}: {RESET}")


	train_dataloader, val_dataloader, test_dataloader, LABELS, label_test = dataset_importer(DATASET, VOCAB_SIZE, MAX_LENGTH, BATCH_SIZE)		
	N_LABELS = len(LABELS)
	########################################################################################
	for _ in range(5):
		print(f"{ATTRIBUTES['Bold']}Models initialization:{RESET}")

		config = MambaConfig(d_model=EMBED_DIM, n_layers=LAYERS, expand_factor=FORWARD_EXPANSION)
		model = Mamba(config)
		cls = Mamba_classifier(model, EMBED_DIM, REDUCED_EMBEDDING_DIM, VOCAB_SIZE, N_LABELS)
		
		
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

		EPOCHS = 5
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
