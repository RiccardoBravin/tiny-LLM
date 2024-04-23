import torch

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


from lib import modelv0
from lib import modelv9


from lib import utils

from lib.dataset import dataset_importer
from lib import utils

VOCAB_SIZE = 512*8
BATCH_SIZE = 64

N_RUNS = 5
EPOCHS = 25
LR = 2e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################################################################################
print("Loading dataset:")

#'Amazon', "imdb", "sst2" "sst5" "twitter" "race" "yelp"
DATASETS = ["Amazon", "imdb", "sst2", "sst5", "twitter", "race", "yelp"]

for dataset in DATASETS:
	MAX_LENGTH = 512

	train_dataloader, test_dataloader, N_LABELS, label_test = dataset_importer(dataset, VOCAB_SIZE, MAX_LENGTH, BATCH_SIZE)

	# ------------ Model 0 ------------
	EMBED_DIM = 128
	NUM_HEADS = 8
	FORWARD_EXPANSION = 2
	LAYERS = 1
	predictions = []
	
	for i in range(N_RUNS):
		
		classifier = modelv0.Classifier(VOCAB_SIZE, MAX_LENGTH, EMBED_DIM, NUM_HEADS, FORWARD_EXPANSION, N_LABELS)
		classifier.to(device)
		
		utils.trainer(classifier, train_dataloader, LR, EPOCHS)

		predictions += utils.evaluator(classifier, test_dataloader)

	print(f"model 0 report for dataset:{dataset}")
	print(classification_report(label_test*N_RUNS, predictions))

	#save model
	t_string = "./models/transformerV0_"  + str(EMBED_DIM) + "_" + str(NUM_HEADS) + "_" + str(FORWARD_EXPANSION) + "_" + str(MAX_LENGTH) + "_" + str(LAYERS) + "_" + str(VOCAB_SIZE) + "_" + str(BATCH_SIZE) + "_" + str(EPOCHS) + "_" + dataset + ".pt"
	torch.save(classifier.state_dict(), t_string)
	
	# ------------ Model 9 ------------
	REDUCED_EMBEDDING_DIM = 16
	EMBED_DIM = 128
	NUM_HEADS = 8
	FORWARD_EXPANSION = 0.1
	LAYERS = 1
	predictions = []
		
	for i in range(N_RUNS):
		classifier = modelv9.Classifier(VOCAB_SIZE, MAX_LENGTH, REDUCED_EMBEDDING_DIM, EMBED_DIM, NUM_HEADS, FORWARD_EXPANSION, LAYERS, N_LABELS)
		classifier.to(device)
		
		utils.trainer(classifier, train_dataloader, LR, EPOCHS)
	
		predictions += utils.evaluator(classifier, test_dataloader)

	print(f"model 9 report for dataset:{dataset}")
	print(classification_report(label_test*N_RUNS, predictions))

	#save model
	t_string = "./models/transformerV9_"  + str(REDUCED_EMBEDDING_DIM) + "_" + str(EMBED_DIM)  + "_" + str(FORWARD_EXPANSION) + "_" + str(MAX_LENGTH) + "_" + str(LAYERS) + "_" + str(VOCAB_SIZE) + "_" + str(BATCH_SIZE) + "_" + str(EPOCHS) + "_" + dataset + ".pt"
	torch.save(classifier.state_dict(), t_string)

	# ------------ Model 9 ------------
	REDUCED_EMBEDDING_DIM = 16
	EMBED_DIM = 128
	NUM_HEADS = 8
	FORWARD_EXPANSION = 2
	LAYERS = 4
	predictions = []
		
	for i in range(N_RUNS):
		classifier = modelv9.Classifier(VOCAB_SIZE, MAX_LENGTH, REDUCED_EMBEDDING_DIM, EMBED_DIM, NUM_HEADS, FORWARD_EXPANSION, LAYERS, N_LABELS)
		classifier.to(device)
		
		utils.trainer(classifier, train_dataloader, LR, EPOCHS)

		predictions += utils.evaluator(classifier, test_dataloader)

	print(f"model 9 report for dataset:{dataset}")
	print(classification_report(label_test*N_RUNS, predictions))

	#save model
	t_string = "./models/transformerV9_"  + str(REDUCED_EMBEDDING_DIM) + "_" + str(EMBED_DIM)  + "_" + str(FORWARD_EXPANSION) + "_" + str(MAX_LENGTH) + "_" + str(LAYERS) + "_" + str(VOCAB_SIZE) + "_" + str(BATCH_SIZE) + "_" + str(EPOCHS) + "_" + dataset + ".pt"
	torch.save(classifier.state_dict(), t_string)

	# ------------ Model 9 ------------
	REDUCED_EMBEDDING_DIM = 16
	EMBED_DIM = 128
	NUM_HEADS = 8
	FORWARD_EXPANSION = 0.1
	LAYERS = 4
	predictions = []
		
	for i in range(N_RUNS):
		classifier = modelv9.Classifier(VOCAB_SIZE, MAX_LENGTH, REDUCED_EMBEDDING_DIM, EMBED_DIM, NUM_HEADS, FORWARD_EXPANSION, LAYERS, N_LABELS)
		classifier.to(device)
		
		utils.trainer(classifier, train_dataloader, LR, EPOCHS)

		predictions += utils.evaluator(classifier, test_dataloader)

	print(f"model 9 report for dataset:{dataset}")
	print(classification_report(label_test*N_RUNS, predictions))

	#save model
	t_string = "./models/transformerV9_"  + str(REDUCED_EMBEDDING_DIM) + "_" + str(EMBED_DIM)  + "_" + str(FORWARD_EXPANSION) + "_" + str(MAX_LENGTH) + "_" + str(LAYERS) + "_" + str(VOCAB_SIZE) + "_" + str(BATCH_SIZE) + "_" + str(EPOCHS) + "_" + dataset + ".pt"
	torch.save(classifier.state_dict(), t_string)


