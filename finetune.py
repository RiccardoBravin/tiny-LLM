# IMPORTS THIRD PARTY MODULES
# ENVIRONMENT VARIABLES
import os
import random

from transformers import Trainer, TrainingArguments

from datasets import Dataset

# IMPORTS CUSTOM MODULES
from lib.colors import ATTRIBUTES, FOREGROUND_COLORS, RESET
from lib.Models.classifiers import (
    PretrainingClassifier,
    RMSClassifier,
    SequenceClassifier,
)
from lib.preprocessing import dataset_selector, load_pretr_tokenizer, make_tokenizer
from lib.utils import (
    CustomPrinterCallback,
    compute_metrics,
    model_size,
    save_model_score,
)
from models_config import *

os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Enables parallelism for tokenizers

# CUSTOM CONSTANTS
TITLE = f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS['BrightYellow']}"
CHECKPOINT = 550000
TRAIN_ITERS = 3

# MODEL CONFIGURATION
# config = EmbBERT_config
config = EmbBERT_Med_config

# Training arguments
training_args = TrainingArguments(
    run_name=f"{config.model_type}_Med_finetuning",  # name of the run
    output_dir="./results",  # output directory
    dataloader_num_workers=6,  # number of dataloader workers (4 works well but might need to be adjusted)
    save_total_limit=1,  # number of total save checkpoints
    overwrite_output_dir=True,  # overwrite the content of the output directory
    eval_strategy="epoch",  # when to evaluate the model
    logging_strategy="epoch",  # log every epoch
    logging_dir=None,  # directory for storing logs
    include_tokens_per_second=False,  # log tokens per second
    include_num_input_tokens_seen=False,  # log number of input tokens seen
    save_strategy="epoch",  # checkpoint save strategy
    load_best_model_at_end=True,  # load the best model when finished training
    metric_for_best_model="mcc",  # use accuracy to evaluate the best model
    num_train_epochs=2,  # total number of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    learning_rate=1e-4,  # learning rate
    lr_scheduler_type="constant",  # learning rate scheduler type
    weight_decay=0.05,  # strength of weight decay
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    eval_delay=8,
    torch_compile=True,
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
    "cola",
    "mrpc",
    "rte",
    "sst2",
    "wnli",
#    "stsb",
    "imdb",
    "news",
    "bull",
    "limit",
#    "nlu",
    "snips",
    "emotion_split",
    "qqp",
    "qnli",
    "mnli-m",
    "mnli-mm",
]


for dataset in datasets:
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
    if CHECKPOINT:
        tokenizer = load_pretr_tokenizer(config.vocab_size, config.max_length)
        print(f"\tLoading tokenizer from checkpoint")
    else:
        print(f"\tCreating new tokenizer/loading dataset's custom one")
        tokenizer = make_tokenizer(
            tokenizer_type="bpe",
            dictionary_size=config.vocab_size,
            dataset_name=dataset,
            train_dataset=train_data,
        )

    print(f"{TITLE}Tokenizing dataset{RESET}")

    def tokenize_batch(batch, tokenizer, max_length):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    # tokenized_train_data = tokenizer(
    #     train_data["text"], truncation=True, padding=True, max_length=config.max_length
    # )

    tokenized_train_data = train_data.map(
        lambda batch: tokenize_batch(batch, tokenizer, config.max_length),
        batched=True,
        batch_size=1000,  # adjust batch size to fit your RAM
    )

    tokenized_test_data = test_data.map(
        lambda batch: tokenize_batch(batch, tokenizer, config.max_length),
        batched=True,
        batch_size=1000,
    )


    tokenized_train_data = tokenized_train_data.add_column("labels", train_data["label"])
    tokenized_test_data = tokenized_test_data.add_column("labels", test_data["label"])

    # Split training into train/validation
    split_dataset = tokenized_train_data.train_test_split(test_size=0.1, shuffle=True)
    train_dataset, validation_dataset = split_dataset["train"], split_dataset["test"]

    test_dataset = tokenized_test_data

    del tokenized_train_data, tokenized_test_data

    best_metric = None
    for count in range(1, TRAIN_ITERS + 1):
        training_args.seed = count * random.randint(1, 1000)
        # TRAINING
        print(f"{TITLE}Initializing model{RESET}")

        if CHECKPOINT:
            print(f"\tLoading model from checkpoint")
            pretr = PretrainingClassifier.from_pretrained(
                f"./results/pretraining/mlm_{config.model_type}_Med/checkpoint-{CHECKPOINT}",
                config=config,
            )

            if (
                config.model_type == "NanoEmbedder"
                or config.model_type == "NanoEmbedderConv"
            ):
                classifier = RMSClassifier(config=config)
                print(f"\tUsing RMS Classifier")
            else:
                classifier = SequenceClassifier(config=config)
                print(
                    f"{FOREGROUND_COLORS['BrightRed']}Using Sequence Classifier{RESET}"
                )

            classifier.change_internal_model(pretr.model)
        else:
            print(
                f"{FOREGROUND_COLORS['BrightRed']}FAILED TO LOAD CHECKPOINT, CHECK CHECKPOINT VARIABLE{RESET}"
            )
            if (
                config.model_type == "NanoEmbedder"
                or config.model_type == "NanoEmbedderConv"
                or config.model_type == "MAMBA"
            ):
                classifier = RMSClassifier(config=config)
                print(f"{FOREGROUND_COLORS['BrightRed']}Using RMS Classifier{RESET}")
            else:
                classifier = SequenceClassifier(config=config)
                print(
                    f"{FOREGROUND_COLORS['BrightRed']}Using Sequence Classifier{RESET}"
                )

        print(
            f"{TITLE}{FOREGROUND_COLORS['BrightMagenta']}Training iteration {count} for dataset {dataset}{RESET}"
        )
        trainer = Trainer(
            model=classifier,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=validation_dataset,  # evaluation dataset
            compute_metrics=compute_metrics,  # the callback that computes metrics of interest
            callbacks=[CustomPrinterCallback],  # custom callback
        )
        print(f"{FOREGROUND_COLORS['BrightGreen']}")
        trainer.train()

        print(f"{TITLE}Evaluating model{RESET}")
        metrics = trainer.evaluate(test_dataset)

        save_model_score(
            metrics, f"./results/finetuning/{config.model_type}_Med/", f"{dataset}.txt"
        )

        if (
            best_metric is None
            or abs(metrics["eval_" + training_args.metric_for_best_model]) > best_metric
        ):
            best_metric = abs(metrics["eval_" + training_args.metric_for_best_model])
            trainer.save_model(
                f"./results/finetuning/{config.model_type}_Med/{dataset}"
            )
