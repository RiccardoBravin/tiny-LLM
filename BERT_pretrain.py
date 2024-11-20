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



pretr_dataset_config = DataConfig(
						dataset_name="bookcorpus",
						dict_size=pow(2, 13),
						tokenizer_type="bpe",
						batch_size=32,
						max_len=256,
						labels=[0,1]
					)

model_config = ModelConfig(
					model_name=None,
					embedding_dimension=90,
					reduced_embedding_dimension=16,
					number_of_heads=2,
					max_length=pretr_dataset_config.max_len,
					forward_expansion=2,
					d_state=None,
					num_layers=2,
					vocab_size=pretr_dataset_config.dict_size,
					learning_rate = 5e-4,
				)



########################################################################################


#load the dataset for pretraining
print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}Importing pretraining dataset (Book Corpus){RESET}")
pretraining_dataset, _ = dataset_selector(pretr_dataset_config.dataset_name, reduced=False)
print(f"{FOREGROUND_COLORS["BrightCyan"]}Dataset contains {len(pretraining_dataset)} training samples to be combined in pairs{RESET}")


#load the tokenizer
print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}Loading/building tokenizer{RESET}")
print(f"{FOREGROUND_COLORS["BrightCyan"]}", end="")
tokenizer = make_tokenizer(pretr_dataset_config, pretraining_dataset)
print(f"{RESET}")


# tokenizing the dataset
print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}Tokenizing dataset{RESET}")
train_dataloader = encode_pretr_dataset(tokenizer, pretraining_dataset, pretr_dataset_config.max_len, pretr_dataset_config.batch_size)
#validation_dataloader = encode_dataset(tokenizer, validation_dataset, dataset_config.max_len, dataset_config.batch_size)
#test_dataloader = encode_dataset(tokenizer, test_dataset, dataset_config.max_len, dataset_config.batch_size)

print(f"\n\n{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}--------------------- STARTING PRETRAINING ---------------------{RESET}\n")

# model = BERT_original(model_config, dropout=0)
# model = Nano_Bert_Efficient(model_config, dropout=0)
# model =  Mamba_model(model_config, dropout=0)
# model = EmbBERT(model_config, dropout=0)
# model = Nano_Bert_Differential_Efficient(model_config, dropout=0)
# model = Nano_Bert_Differential_Skip(model_config, dropout=0)
model = NanoBERT_original(model_config, dropout=0)

model_config.model_name = model.__class__.__name__

# cls = Classifier_BERT_pretraining(model, model_config.embedding_dimension, pretr_dataset_config.dict_size, pretr_dataset_config.n_labels())
cls = Classifier_Nano_BERT_pretraining(model, model_config.embedding_dimension, model_config.reduced_embedding_dimension, pretr_dataset_config.dict_size, pretr_dataset_config.n_labels())
cls.to(device)


try:
	checkpoint_number = -1 # set to -1 to start from scratch
	resume(cls, f"trained_models/checkpoints/{model.__class__.__name__}_{checkpoint_number}.pth")
	print(f"Loading checkpoint {model.__class__.__name__}_{checkpoint_number}")
	

except:
	print(f"{FOREGROUND_COLORS["BrightRed"]}Checkpoint not found{RESET}")


print(f"Model {model.__class__.__name__} initialized on {device}")

print(f"{ATTRIBUTES['Bold']}Model parameters:{RESET}")
print_model_params(model)


print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightGreen"]}Starting training{RESET}")
trainer = BertTrainer(cls, device, model_config, pretr_dataset_config, checkpoint=checkpoint_number+1)
trainer.train(train_dataloader, epochs_pretraining, log_t_interval=log_t_interval)



