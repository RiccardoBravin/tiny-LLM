# IMPORTS THIRD PARTY MODULES
from datasets import Dataset
from transformers import Trainer, TrainingArguments


# IMPORTS CUSTOM MODULES
from lib.colors import RESET, ATTRIBUTES, FOREGROUND_COLORS
from lib.Models.classifiers import SequenceClassifier, PretrainingClassifier, RMSClassifier
from lib.preprocessing import dataset_selector, make_tokenizer
from lib.utils import model_size, compute_metrics, CustomPrinterCallback
from models_config import *

# ENVIRONMENT VARIABLES
import os   
os.environ["TOKENIZERS_PARALLELISM"] = "true" # Enables parallelism for tokenizers

# CUSTOM CONSTANTS
TITLE = f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS['BrightYellow']}"
CHECKPOINT = 461000

# MODEL CONFIGURATION
config = EmbBERT_config

# Training arguments
training_args = TrainingArguments(
    run_name=f"{config.model_type}_finetuning", # name of the run
    
    output_dir='./results',             # output directory
    dataloader_num_workers=4,           # number of dataloader workers (4 works well but might need to be adjusted)
    save_total_limit=1,                 # number of total save checkpoints
    overwrite_output_dir=True,	        # overwrite the content of the output directory
    eval_strategy="epoch",              # when to evaluate the model
    logging_strategy="epoch",           # log every epoch
    logging_dir=None,                   # directory for storing logs
    include_tokens_per_second=False,    # log tokens per second
    include_num_input_tokens_seen=False,# log number of input tokens seen

	save_strategy="epoch",              # checkpoint save strategy
    load_best_model_at_end=True,        # load the best model when finished training 
    metric_for_best_model="mcc",        # use accuracy to evaluate the best model
    
	num_train_epochs=20,                # total number of training epochs
	per_device_train_batch_size=32,     # batch size per device during training
	per_device_eval_batch_size=64,      # batch size for evaluation
    
	learning_rate=5e-5,                 # learning rate
    lr_scheduler_type="constant",       # learning rate scheduler type
	weight_decay=0.05,                  # strength of weight decay


)

# PRINTING MODEL CONFIGURATION
print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS['BrightYellow']}Model configs{RESET}")
print(config)


print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS['BrightYellow']}Model checkup{RESET}")
aux = SequenceClassifier(config=config)
print(aux.model)

print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS['BrightYellow']}Model size{RESET}")
print(model_size(aux.model))

del aux


# DATASETS SELECTION
datasets = [
    # "cola", 
    # "mrpc", 
    # "qnli", 
    # "qqp", 
    # "rte", 
    # "sst2", 
    # "wnli", 
    # "stsb", 
    "imdb", 
    "news", 
    "bull", 
    "limit", 
    "nlu", 
    "snips", 
    "emotion_split", 
    "mnli-m", 
    "mnli-mm"
]


for dataset in datasets:

    print(f"{TITLE}Loading dataset {dataset}{RESET}")
    train_data, test_data = dataset_selector(dataset)

    if dataset != "stsb":
        config.num_labels = len(train_data.unique("label"))
        training_args.metric_for_best_model = "mcc"
    else:
        config.num_labels = 1
        training_args.metric_for_best_model = "scc"

    print(f"{TITLE}Training/Loading tokenizer{RESET}")
    tokenizer = make_tokenizer(tokenizer_type="bpe", dictionary_size=config.vocab_size, dataset_name=dataset, train_dataset=train_data)

    print(f"{TITLE}Tokenizing dataset{RESET}")
    tokenized_train_data = tokenizer(train_data['text'], truncation=True, padding=True, max_length=config.max_length)

    train_dataset = Dataset.from_dict({
        'input_ids': tokenized_train_data['input_ids'],
        'attention_mask': tokenized_train_data['attention_mask'],
        'labels': train_data['label']
    })
    
    # Splitting the dataset
    validation_dataset = train_dataset.train_test_split(test_size=0.1)
    train_dataset, validation_dataset = validation_dataset["train"], validation_dataset["test"]


    tokenized_test_data = tokenizer(test_data['text'], truncation=True, padding=True, max_length=256)

    test_dataset = Dataset.from_dict({
        'input_ids': tokenized_test_data['input_ids'],
        'attention_mask': tokenized_test_data['attention_mask'],
        'labels': test_data['label']
    })


    for count in range(1, 6):

        # TRAINING
        print(f"{TITLE}Initializing model{RESET}")

        if CHECKPOINT:
            print(f"{FOREGROUND_COLORS['BrightRed']}Loading model from checkpoint{RESET}")
            pretr = PretrainingClassifier.from_pretrained(f"./results/mlm_{config.model_type}/checkpoint-{CHECKPOINT}", config=config)
            
            if config.model_type == "NanoEmbedder" or config.model_type == "NanoEmbedderConv":
                classifier = RMSClassifier(config=config)
                print(f"{FOREGROUND_COLORS['BrightRed']}Using RMS Classifier{RESET}")
            else:
                classifier = SequenceClassifier(config=config)
                print(f"{FOREGROUND_COLORS['BrightRed']}Using Sequence Classifier{RESET}")
            
            
            classifier.change_internal_model(pretr.model)
        else:
            if config.model_type == "NanoEmbedder" or config.model_type == "NanoEmbedderConv" or config.model_type == "MAMBA":
                classifier = RMSClassifier(config=config)
                print(f"{FOREGROUND_COLORS['BrightRed']}Using RMS Classifier{RESET}")
            else:
                classifier = SequenceClassifier(config=config)
                print(f"{FOREGROUND_COLORS['BrightRed']}Using Sequence Classifier{RESET}")




        print(f"{TITLE}{FOREGROUND_COLORS['BrightMagenta']}Training iteration {count}")
        trainer = Trainer(
        	model=classifier,               		    # the instantiated 🤗 Transformers model to be trained
        	args=training_args,             		    # training arguments, defined above

            train_dataset=train_dataset,    		    # training dataset
        	eval_dataset=validation_dataset,            # evaluation dataset

        	compute_metrics=compute_metrics,			# the callback that computes metrics of interest
            callbacks=[CustomPrinterCallback]                # custom callback
        )

        trainer.train()

        print(f"{TITLE}Evaluating model{RESET}")
        metrics = trainer.evaluate(test_dataset)











