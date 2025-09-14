# IMPORTS THIRD PARTY MODULES
from datasets import Dataset
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
import random

# IMPORTS CUSTOM MODULES
from lib.colors import RESET, ATTRIBUTES, FOREGROUND_COLORS
from lib.Models.classifiers import SequenceClassifier, RMSClassifier
from lib.preprocessing import dataset_selector, make_tokenizer, load_pretr_tokenizer
from lib.utils import model_size, compute_metrics, CustomPrinterCallback, save_model_score
from models_config import *

# ENVIRONMENT VARIABLES
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true" # Enables parallelism for tokenizers


# SUPPRESS WARNINGS
import warnings
# Suppress the specific warning
warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization")


# CUSTOM CONSTANTS
TITLE = f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS['BrightYellow']}"
TRAIN_ITERS = 5



# MODEL CONFIGURATION
# config = EmbBERT_config
config = EmbBERT_Tiny_config

# Training arguments
training_args = TrainingArguments(
    run_name=f"{config.model_type}_tiny_quantized", # name of the run

    output_dir="./results/quantization", # output directory

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

	num_train_epochs=2,                # total number of training epochs
	per_device_train_batch_size=32,     # batch size per device during training
	per_device_eval_batch_size=64,      # batch size for evaluation

    optim="adamw_bnb_8bit",
	learning_rate=1e-4,                 # learning rate
    lr_scheduler_type="constant",       # learning rate scheduler type
	weight_decay=0.05,                  # strength of weight decay

    label_names=["labels"],
    fp16=True,

    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    eval_delay=4,
    torch_compile=True,

)

# PRINTING MODEL CONFIGURATION
print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS['BrightYellow']}Model configs{RESET}")
print(config)



# DATASETS SELECTION
datasets = [
    "cola",
    "mrpc",
    "rte",
    "sst2",
    "wnli",
    "stsb",
    "imdb",
    "news",
    "bull",
    "limit",
    "nlu",
    "snips",
    "emotion_split",
    "qqp",
    "qnli",
    "mnli-m",
    "mnli-mm"
]



for dataset in datasets:
    training_args.seed = random.randint(0, 1000)

    print(f"{TITLE}Loading dataset {dataset}{RESET}")
    train_data, test_data = dataset_selector(dataset)

    if dataset != "stsb":
        config.num_labels = len(train_data.unique("label"))
        training_args.metric_for_best_model = "mcc"
        training_args.greater_is_better = True
        if dataset == "wnli":
            training_args.greater_is_better = False
    else:
        config.num_labels = 1
        training_args.metric_for_best_model = "scc"
        training_args.greater_is_better = True

    print(f"{TITLE}Training/Loading tokenizer{RESET}")
    tokenizer = load_pretr_tokenizer(config.vocab_size, config.max_length)

    #tokenizer = make_tokenizer(tokenizer_type="bpe", dictionary_size=config.vocab_size, dataset_name=dataset, train_dataset=train_data)

    print(f"{TITLE}Tokenizing dataset{RESET}")
    tokenized_train_data = tokenizer(train_data['text'], truncation=True, padding=True, max_length=config.max_length)

    train_dataset = Dataset.from_dict({
        'input_ids': tokenized_train_data['input_ids'],
        'attention_mask': tokenized_train_data['attention_mask'],
        'labels': train_data['label']
    })

    # Splitting the dataset
    validation_dataset = train_dataset.train_test_split(test_size=0.1).shuffle()
    train_dataset, validation_dataset = validation_dataset["train"], validation_dataset["test"]


    tokenized_test_data = tokenizer(test_data['text'], truncation=True, padding=True, max_length=256)

    test_dataset = Dataset.from_dict({
        'input_ids': tokenized_test_data['input_ids'],
        'attention_mask': tokenized_test_data['attention_mask'],
        'labels': test_data['label']
    })




    # TRAINING
    print(f"{TITLE}Initializing model{RESET}")

    q_conf = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )

    print(f"\tLoading model from checkpoint{RESET}")
    try:
        if config.model_type == "NanoEmbedder" or config.model_type == "NanoEmbedderConv":
            print(f"{FOREGROUND_COLORS['BrightRed']}Using RMS Classifier{RESET}")
            classifier = RMSClassifier(config=config)

        else:
            print(f"{FOREGROUND_COLORS['BrightRed']}Using Sequence Classifier{RESET}")
            classifier = SequenceClassifier.from_pretrained(
                                                    f"./results/finetuning/{config.model_type}_tiny/{dataset}",
                                                    config=config,
                                                    quantization_config = q_conf,
                                            )
        print(classifier)
        print(f"Model size: {classifier.get_memory_footprint()/1000}KB")

    except:
        print(f"{FOREGROUND_COLORS['BrightRed']}FAILED TO LOAD CHECKPOINT, CHECK CHECKPOINT VARIABLE{RESET}")
        exit()




    print(f"{TITLE}Training model{RESET}")

    from peft import LoraConfig, get_peft_model
    peft_config = LoraConfig(
        target_modules="all-linear",
    )
    classifier = get_peft_model(classifier, peft_config)
    classifier.print_trainable_parameters()
    trainer = Trainer(
        model=classifier,               		    # the instantiated 🤗 Transformers model to be trained
        args=training_args,             		    # training arguments, defined above

        train_dataset=train_dataset,      		    # training dataset
        eval_dataset=validation_dataset,                  # evaluation dataset

        compute_metrics=compute_metrics,			# the callback that computes metrics of interest
        callbacks=[CustomPrinterCallback]           # custom callback
    )

    trainer.train()


    print(f"{TITLE}Evaluating model{RESET}")
    metrics = trainer.evaluate(test_dataset)
    save_model_score(metrics, f"./results/quantization/{config.model_type}_tiny/", f"{dataset}.txt")











