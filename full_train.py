import torch

import evaluate


from lib import utils
from lib.MAMBA import Mamba, MambaConfig

from lib.dataset import dataset_importer

VOCAB_SIZE = 512*8
BATCH_SIZE = 32

REDUCED_EMBEDDING_DIM = 16
EMBED_DIM = 128
NUM_HEADS = 8
FORWARD_EXPANSION = 4
MAX_LENGTH = 512
LAYERS = 1


class Mamba_classifier(torch.nn.Module):
		def __init__(self, model, d_model, reduced_d_model, vocab_size, n_labels):
			super(Mamba_classifier, self).__init__()
			self.embedder = torch.nn.Embedding(vocab_size, reduced_d_model)
			self.embed_expander = torch.nn.Linear(reduced_d_model, d_model)
			self.model = model
			self.fc = torch.nn.Linear(d_model, n_labels)

		def forward(self, x):
			embedded = self.embedder(x)
			embedded = self.embed_expander(embedded)
			processed = self.model(embedded)
			x_mean = processed.mean(dim=1)
			out = self.fc(x_mean)
			return out
		
#initialize model
config = MambaConfig(
	d_model=EMBED_DIM,
	n_layers=LAYERS
)

########################################################################################
#'Amazon', "imdb", "sst2" "sst5" "twitter" "race" "yelp" "news" "trec_coarse" "bull"

for DATASET in ["imdb", "sst2", "news", "trec_coarse", "bull"]:
	print(f"Loading dataset {DATASET}:")


	train_dataloader, val_dataloader, test_dataloader, N_LABELS, label_test = dataset_importer(DATASET, VOCAB_SIZE, MAX_LENGTH, BATCH_SIZE)		

	########################################################################################
	for _ in range(5):
		print("Model initialization:")

		model = Mamba(config)

		classifier = Mamba_classifier(model, EMBED_DIM, REDUCED_EMBEDDING_DIM, VOCAB_SIZE, N_LABELS)

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
		print("Starting training")

		EPOCHS = 5
		LR = 1e-2
	
		
		utils.trainer(classifier, train_dataloader, val_dataloader, LR, EPOCHS)
		
		########################################################################################
		print("Starting evaluation")
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
			f.write(f"average eval loss: {avg_eval_loss}\n")
			f.write(f"Accuracy: {accuracy['accuracy']}\n")
			f.write(f"F1: {f1['f1']}\n")
			f.write(f"Precision: {precision['precision']}\n")
			f.write(f"Recall: {recall['recall']}\n")
			f.write(f"MCC: {mcc['matthews_correlation']}\n")
			f.write(f"Confusion matrix:\n {conf_mat['confusion_matrix']}\n")
			f.write(str(utils.model_size(classifier)))
			f.write("\n\n*******************************************\n\n")

	with open(f"results/{DATASET}_classification_report.txt", "a") as f:
			f.write(f"**************************************************\n==================================================\n**************************************************\n")
