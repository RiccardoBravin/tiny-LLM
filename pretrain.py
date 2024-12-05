# IMPORTS THIRD PARTY MODULES
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import Trainer, TrainingArguments


# IMPORTS CUSTOM MODULES
from lib.colors import RESET, ATTRIBUTES, FOREGROUND_COLORS
from lib.Models.EmbBERT import EmbBERT
from lib.Models.classifiers import PretrainingClassifier
from lib.preprocessing import pretr_tokenizer, pretr_dataset_builder
from lib.utils import model_size, CustomPrinterCallback, CustomLoggerCallback
from models_config import *

# ENVIRONMENT VARIABLES
import os   
os.environ["TOKENIZERS_PARALLELISM"] = "true" # Enables parallelism for tokenizers

# CUSTOM CONSTANTS
TITLE = f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS['BrightYellow']}"
DEBUG = True


# MODEL CONFIGURATION
config = EmbBERT_config

# Training arguments
training_args = TrainingArguments(
    run_name=f"{config.model_type}_pretraining", 
    output_dir=f'./results/mlm_{config.model_type}',

    dataloader_num_workers=4,           # number of dataloader workers (4 works well but might need to be adjusted)
    save_total_limit=5,                 # number of total save checkpoints
    overwrite_output_dir=True,	        # overwrite the content of the output directory
    eval_strategy="steps",              # when to evaluate the model
    logging_strategy="steps",           # log every epoch
    logging_steps=1000,                 # log every 1000 steps
    eval_steps=1000,                    # evaluate every 1000 steps
    logging_dir=None,                   # directory for storing logs
    include_tokens_per_second=False,    # log tokens per second
    include_num_input_tokens_seen=False,# log number of input tokens seen

	save_strategy="steps",               # checkpoint save strategy
    save_steps=1000,                    # save checkpoint every 1000 steps
    load_best_model_at_end=True,        # load the best model when finished training 
    metric_for_best_model="loss",       # use accuracy to evaluate the best model
    
	num_train_epochs=1,                # total number of training epochs
	per_device_train_batch_size=32,     # batch size per device during training
	per_device_eval_batch_size=64,      # batch size for evaluation
    
	learning_rate=2e-4,                 # learning rate
    lr_scheduler_type="constant",       # learning rate scheduler type
	weight_decay=0.05,                  # strength of weight decay

)

# PRINTING MODEL CONFIGURATION
print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS['BrightYellow']}Model configs{RESET}")
print(config)

print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS['BrightYellow']}Model checkup{RESET}")
print(EmbBERT(config))

print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS['BrightYellow']}Model size{RESET}")
print(model_size(EmbBERT(config)))




print(f"{TITLE}Loading dataset Book Corpus{RESET}")

dataset = load_dataset("bookcorpus/bookcorpus", cache_dir="./datasets", trust_remote_code=True)
if DEBUG:
    train_data = dataset['train'].select(range(0, 1000000))
else:
    train_data = dataset['train']
del dataset
print(f"\tLoaded dataset of size: {len(train_data)}")

print(f"{TITLE}Training/Loading tokenizer{RESET}")
tokenizer = pretr_tokenizer(train_data, config.vocab_size)

print(f"{TITLE}Preprocessing dataset{RESET}")
train_data, eval_data = pretr_dataset_builder(train_data, tokenizer, config.max_length)



# Masked language modeling collator
from transformers import DataCollatorForLanguageModeling
mlm_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)


# TRAINING
print(f"{TITLE}Initializing model{RESET}")
model = EmbBERT(config)
classifier = PretrainingClassifier(config=config)


print(f"{TITLE}{FOREGROUND_COLORS['BrightMagenta']}Pretraining model")
trainer = Trainer(
    model=classifier,               		# the instantiated 🤗 Transformers model to be trained
    args=training_args,             		# training arguments, defined above

    train_dataset=train_data,    		    # training dataset
    eval_dataset=eval_data,                 # evaluation dataset
    
    data_collator=mlm_collator,             # data collator

    callbacks=[CustomLoggerCallback, CustomPrinterCallback]       # custom callback
)

trainer.train()





