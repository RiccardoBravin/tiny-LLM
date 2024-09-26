
#load 

#COLORS
from colors import ATTRIBUTES, FOREGROUND_COLORS, RESET

#STD
import os
import torch
from tqdm import tqdm


#CUSTOM
from lib.configs import DataConfig, ModelConfig
from lib.utils import model_size, print_model_params, calculate_metrics, metrics_to_str, resume
from lib.preprocessing import dataset_selector, make_tokenizer, encode_dataset, encode_pretr_dataset

from lib.trainer import BertTrainer, Trainer

from lib.Models.final_classifiers import *
from lib.Models.models import *


epochs_pretraining = 1

log_t_interval = 60

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



dataset_config = DataConfig(
						dataset_name="bookcorpus",
						dict_size=pow(2, 13),
						tokenizer_type="bpe",
						batch_size=32,
						max_len=256,
						labels=[0,1]
					)

model_config = ModelConfig(
					model_name=None,
					embedding_dimension=48,
					reduced_embedding_dimension=16,
					number_of_heads=0,
					max_length=dataset_config.max_len,
					forward_expansion=2,
					d_state=4,
					num_layers=4,
					vocab_size=dataset_config.dict_size,
					learning_rate = 5e-4,
				)



########################################################################################


#load the dataset for pretraining
print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}Importing pretraining dataset (Book Corpus){RESET}")
pretraining_dataset, _ = dataset_selector(dataset_config.dataset_name, reduced=True)
print(f"{FOREGROUND_COLORS["BrightCyan"]}Dataset contains {len(pretraining_dataset)} training samples to be combined in pairs{RESET}")


#load the tokenizer
print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}Loading/building tokenizer{RESET}")
print(f"{FOREGROUND_COLORS["BrightCyan"]}", end="")
tokenizer = make_tokenizer(dataset_config, pretraining_dataset)
print(f"{RESET}")


# tokenizing the dataset
print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}Tokenizing dataset{RESET}")
train_dataloader = encode_pretr_dataset(tokenizer, pretraining_dataset, dataset_config.max_len, dataset_config.batch_size)

print(f"\n\n{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}--------------------- STARTING PRETRAINING ---------------------{RESET}\n")

# model = BERT_original(model_config, dropout=0)
# model = Nano_Bert_Efficient(model_config, dropout=0)
# model = Embedder_model(model_config, dropout=0)
# model = Embedder_conv_model(model_config, dropout=0)
model = Mamba_model(model_config, dropout=0)

model_config.model_name = model.__class__.__name__		

print(f"{ATTRIBUTES['Bold']}Model parameters:{RESET}")
print_model_params(model)

classifier = Classifier_Nano_BERT_pretraining(model, model_config.embedding_dimension, model_config.reduced_embedding_dimension, dataset_config.dict_size, dataset_config.n_labels())
# classifier = Classifier_BERT_pretraining(model, model_config.embedding_dimension, dataset_config.dict_size, dataset_config.n_labels())
classifier.to(device)
print(f"{RESET}Model {model.__class__.__name__} initialized on {device}")


#cycle over each model in the folder trained_models/checkpoints that has the same name as the model
for checkpoint in os.listdir("trained_models/checkpoints"):
	if model.__class__.__name__ in checkpoint:
		try:
			print("----------------------------------------------------------------------------------------")

			print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS['BrightMagenta']}Loading checkpoint {checkpoint}")
			resume(classifier, f"trained_models/checkpoints/{checkpoint}")


			print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightGreen"]}Starting training{RESET}")
			trainer = BertTrainer(classifier, device, model_config, dataset_config)
			trainer.evaluate(train_dataloader)
		except Exception as e:
			print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightRed"]}Error in checkpoint {checkpoint}: {e}{RESET}")
			continue


