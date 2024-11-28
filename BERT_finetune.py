#COLORS
from colors import ATTRIBUTES, FOREGROUND_COLORS, RESET

#STD
import os
import torch
from tqdm import tqdm


#CUSTOM
from lib.configs import DataConfig, ModelConfig
from lib.utils import model_size, print_model_params, calculate_metrics, metrics_to_str, resume
from lib.preprocessing import dataset_selector, make_tokenizer, load_tokenizer, encode_dataset, encode_pretr_dataset

from lib.trainer import BertTrainer, Trainer

from lib.Models.final_classifiers import *
from lib.Models.models import *


epochs_training = 10
min_iterations = 2000
finetuning_tests= 5

logs_x_epoch = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataset_config = DataConfig(
						dataset_name=None,
						dict_size=pow(2, 11),
						tokenizer_type="bpe",
						batch_size=32,
						max_len=256,
						labels=[0,1]
					)

model_config = ModelConfig(
					model_name=None,
					embedding_dimension=64,
					reduced_embedding_dimension=None,
					number_of_heads=None,
					max_length=dataset_config.max_len,
					forward_expansion=1,
					d_state=6,
					num_layers=5,
					vocab_size=dataset_config.dict_size,
					learning_rate = 3e-4,
				)



########################################################################################


# for DATASET_NAME in ["news", "bull", "limit", "nlu", "snips", "imdb", "emotion_split"]: #extra
# for DATASET_NAME in ["cola", "mnli-m", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "wnli", "stsb"]: #GLUE
for DATASET_NAME in ["cola", "mrpc", "qnli", "qqp", "rte", "sst2", "wnli", "stsb", "imdb", "news", "bull", "limit", "nlu", "snips", "emotion_split", "mnli-m", "mnli-mm"]:  #ALL DATASETS
# for DATASET_NAME in ["qqp"]: #GLUE


	dataset_config.dataset_name = DATASET_NAME

	print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}Loading dataset {DATASET_NAME} {RESET}")

	print(f"{FOREGROUND_COLORS["BrightCyan"]}", end="")
	train_dataset, test_dataset = dataset_selector(dataset_config.dataset_name, reduced=False)
	
	if DATASET_NAME != "stsb":
		dataset_config.labels = train_dataset.unique("label")
	else:
		dataset_config.labels = [0]

	print(f"Dataset labels: {dataset_config.labels}")

	#tokenizer = make_tokenizer(dataset_config, train_dataset)
	tokenizer = load_tokenizer(dataset_config, f"{dataset_config.tokenizer_type}_bookcorpus_{dataset_config.dict_size}") 

	validation_dataset = train_dataset.train_test_split(test_size=0.1)
	train_dataset, validation_dataset = validation_dataset["train"], validation_dataset["test"]

	train_dataloader = encode_dataset(tokenizer, train_dataset, dataset_config.max_len, dataset_config.batch_size, shuffle=True)
	validation_dataloader = encode_dataset(tokenizer, validation_dataset, dataset_config.max_len, dataset_config.batch_size, shuffle=False)
	test_dataloader = encode_dataset(tokenizer, test_dataset, dataset_config.max_len, dataset_config.batch_size, shuffle=False)

	train_n = 0
	while train_n < finetuning_tests:

		print(f"\n\n{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}--------------------- STARTING TRAINING CYCLE {train_n} ---------------------{RESET}\n")


		########################################################################################
		print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightGreen"]}Initializing model{RESET}")
		
		# model = BERT_original(model_config, dropout=0.1)
		# model = Nano_Bert_Efficient(model_config, dropout=0.1)
		# model = Nano_Bert_Efficient_mh(model_config, dropout=0.1)
		# model = Embedder_model(model_config, dropout=0.1)
		# model = Embedder_conv_model(model_config, dropout=0.1)
		# model = Mamba_model(model_config, dropout=0.1)
		# model = Nano_Bert_Efficient_mh(model_config, dropout=0.1)
		# model = Mamba_model(model_config, dropout=0.1)
		# model = Nano_Bert_Differential_Skip(model_config, dropout=0.1)
		# model = NanoBERT_original(model_config, dropout=0.1)
		model = Mamba_model_noNANO(model_config, dropout=0.1)

		model_config.model_name = model.__class__.__name__	

		try:
			full_model = Classifier_BERT_pretraining(model, model_config.embedding_dimension, dataset_config.dict_size, dataset_config.n_labels())
			# full_model = Classifier_Nano_BERT_pretraining(model, model_config.embedding_dimension, model_config.reduced_embedding_dimension, dataset_config.dict_size, 2)

			checkpoint = f"{model.__class__.__name__}_32"   #CHANGE HERE THE CHECKPOINT TO USE <-----------------------
			print(f"Loading checkpoint {checkpoint}")
			resume(full_model, f"trained_models/checkpoints/{checkpoint}.pth")

			model = full_model.model

		except Exception as e:
			print(e)
			print(f"{FOREGROUND_COLORS["BrightRed"]}Checkpoint not found{RESET}")
			exit()

		#Freezing the model
		# for param in model.embedder.parameters(): 
		# 	param.requires_grad = False

		########################################################################################
		print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS['BrightMagenta']}Normal model initialization:{RESET}")
		# classifier = Classifier_first_token(model, model_config.embedding_dimension, dataset_config.n_labels())
		classifier = Classifier_rms(model, model_config.embedding_dimension, dataset_config.n_labels())
		classifier.to(device)
		print(f"Model {model.__class__.__name__} initialized on {device}")

		model_config.model_name = classifier.__class__.__name__ + "_" + model_config.model_name


		print("Model parameters:")
		print_model_params(classifier)

		########################################################################################

		print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS['BrightMagenta']}Starting training{RESET}")
		trainer = Trainer(classifier, device, model_config)
		trainer.train(train_dataloader, validation_dataloader, epochs_training, log_freq=logs_x_epoch, color=FOREGROUND_COLORS["BrightMagenta"], min_iter=min_iterations)


		########################################################################################

		print(f"{FOREGROUND_COLORS["BrightYellow"]}Testing the model{RESET}")
		print(f"{FOREGROUND_COLORS["BrightCyan"]}", end="")
		predicted, avg_eval_loss = trainer.evaluate(test_dataloader)
		print(f"{RESET}")


		# Evaluating the results
		print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}Evaluating the results{RESET}")
		print(f"{FOREGROUND_COLORS["BrightCyan"]}", end="")
		metrics = calculate_metrics(test_dataloader, predicted)
		print(metrics_to_str(metrics))
		print(f"{RESET}")


		folder_name = f"mlm_{checkpoint}"
		if not os.path.exists(f"results/{folder_name}/"):
			os.makedirs(f"results/{folder_name}/")

		#save the classification report in a file for later use specifying the dataset, model hyperparameters
		with open(f"results/{folder_name}/{dataset_config.dataset_name}_{dataset_config.dict_size}_{dataset_config.tokenizer_type}_pretraining.txt", "a") as f:
			f.write(f"{model_config}\n")
			f.write(f"Trained for {epochs_training} epochs\n\n")
			f.write(f"average eval loss: {avg_eval_loss: .4f}\n")
			f.write(f"{metrics_to_str(metrics)}\n")
			f.write(str(model_size(classifier)))
			f.write("\n\n*******************************************\n\n")

		train_n += 1
