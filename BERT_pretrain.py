#COLORS
from colors import ATTRIBUTES, FOREGROUND_COLORS, RESET

#STD
import os
import torch
from tqdm import tqdm


#CUSTOM
from lib.configs import DataConfig, ModelConfig
from lib.utils import model_size, print_model_params, trainer, evaluator, calculate_metrics, metrics_to_str
from lib.preprocessing import dataset_selector, make_tokenizer, encode_dataset
from lib.Models.final_classifiers import *
from lib.Models.models import *


from lib.trainer import BertTrainer, Trainer

epochs_pretraining = 30
lr_pretraining = 5e-4

epochs_post = 10
lr_post = 1e-3
logs_x_epoch = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataset_config = DataConfig(
					dataset_name=None, 
					dict_size=pow(2, 12), 
					tokenizer_type="bpe", 
					batch_size=128, 
					max_len=256, 
					labels=None
				)

model_config = ModelConfig(
					model_name=None, 
					embedding_dimension=64, 
					reduced_embedding_dimension=16, 
					number_of_heads=8, 
					max_length=dataset_config.max_len, 
					forward_expansion=0.5, 
					num_layers=6,
					vocab_size=dataset_config.dict_size
				)


########################################################################################

for DATASET_NAME in ["imdb","sst2", "news", "bull", "limit", "nlu", "snips", "mnli", "emotion_split"]:
	dataset_config.dataset_name = DATASET_NAME

	#load the dataset   
	print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}Importing dataset {DATASET_NAME}{RESET}")
	train_dataset, test_dataset = dataset_selector(dataset_config.dataset_name)
	dataset_config.labels = train_dataset.unique("label")
	print(f"{FOREGROUND_COLORS["BrightCyan"]}Dataset contains {len(train_dataset)} training samples and {len(test_dataset)} test samples with {dataset_config.labels} labels{RESET}")


	#load the tokenizer
	print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}Loading/building tokenizer{RESET}")
	print(f"{FOREGROUND_COLORS["BrightCyan"]}", end="")
	tokenizer = make_tokenizer(dataset_config, train_dataset)
	print(f"{RESET}")

	#split the training to have a small validation set
	validation_dataset = train_dataset.train_test_split(test_size=0.1)
	train_dataset, validation_dataset = validation_dataset["train"], validation_dataset["test"]

	# tokenizing the dataset
	print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}Tokenizing dataset{RESET}")
	train_dataloader = encode_dataset(tokenizer, train_dataset, dataset_config.max_len, dataset_config.batch_size)
	validation_dataloader = encode_dataset(tokenizer, validation_dataset, dataset_config.max_len, dataset_config.batch_size)
	test_dataloader = encode_dataset(tokenizer, test_dataset, dataset_config.max_len, dataset_config.batch_size)

	train_dataloader.shuffle = True

	train_n = 0
	while train_n < 2:
		
		print(f"\n\n{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}--------------------- STARTING TRAINING CYCLE {train_n} ---------------------{RESET}\n")


		########################################################################################
		print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightGreen"]}Initializing model{RESET}")
		#model = Bert_efficient(model_config)
		model = Nano_Bert_Efficient(model_config)
		# model = Embedder_model(model_config)
		# model = Mamba_model(model_config)
		# model = Embedder_conv_movel(model_config)
		# model = Embbert(model_config)

		model_config.model_name = model.__class__.__name__
		

		# cls = Classifier_BERT_pretraining(model, model_config.embedding_dimension, dataset_config.dict_size, dataset_config.n_labels())
		cls = Classifier_Nano_BERT_pretraining(model, model_config.embedding_dimension, model_config.reduced_embedding_dimension, dataset_config.dict_size, dataset_config.n_labels())
		cls.to(device)
		print(f"Model {model.__class__.__name__} initialized on {device}")


		print(f"{ATTRIBUTES['Bold']}Model parameters:{RESET}")
		print_model_params(model)



		########################################################################################
		print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightGreen"]}Starting training{RESET}")
		trainer = BertTrainer(cls, device, lr_pretraining, model_config, dataset_config)
		trainer.train(train_dataloader, validation_dataloader, epochs_pretraining, log_freq=logs_x_epoch)


		print(f"{RESET}")	






		
		########################################################################################
		print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS['BrightMagenta']}Normal model initialization:{RESET}")
		classifier = Classifier_BERT_post(model, model_config.embedding_dimension, dataset_config.n_labels())
		# classifier = Classifier_rms(model, model_config.embedding_dimension, dataset_config.n_labels())
		classifier.to(device)
		print(f"Model {model.__class__.__name__} initialized on {device}")


		print("Model parameters:")
		print_model_params(classifier)

		########################################################################################

		print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS['BrightMagenta']}Starting training{RESET}")
		trainer = Trainer(classifier, device, lr_post, model_config)
		trainer.train(train_dataloader, validation_dataloader, epochs_post, log_freq=logs_x_epoch, color=FOREGROUND_COLORS["BrightMagenta"])


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


		folder_name = f"mlm_{model_config.model_name}"
		if not os.path.exists(f"results/{folder_name}/"):
			os.makedirs(f"results/{folder_name}/")

		#save the classification report in a file for later use specifying the dataset, model hyperparameters
		with open(f"results/{folder_name}/{dataset_config.dataset_name}_{dataset_config.dict_size}_{dataset_config.tokenizer_type}_pretr_report.txt", "a") as f:
			f.write(f"{model_config}\n")
			f.write(f"LR: {lr_post}\n")
			f.write(f"EPOCHS: {epochs_post}\n\n")
			f.write(f"average eval loss: {avg_eval_loss: .4f}\n")
			f.write(f"{metrics_to_str(metrics)}\n")
			f.write(str(model_size(classifier)))
			f.write("\n\n*******************************************\n\n")

		train_n += 1
