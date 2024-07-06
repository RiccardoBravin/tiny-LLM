from colors import RESET, ATTRIBUTES, FOREGROUND_COLORS, BACKGROUND_COLORS

import torch
import os

from lib.configs import DataConfig, ModelConfig
from lib.utils import model_size, print_model_params, trainer, evaluator, calculate_metrics, metrics_to_str
from lib.preprocessing import dataset_selector, make_tokenizer, encode_dataset
from lib.Models.models import *
from lib.Models.final_classifiers import Classifier_rms, Classifier_BERT, Smart_classifier, Tester_classifier

from lib.trainer import Trainer

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


lr = 5e-3
epochs = 20

dataset_config = DataConfig(
                    dataset_name="qnli", 
                    dict_size=pow(2, 12), 
                    tokenizer_type="bpe", #wordpiece #bpe #unigram 
                    batch_size=64, 
                    max_len=512, 
                    labels=None
                )

model_config = ModelConfig(
                    model_name=None, 
                    embedding_dimension=128, 
                    reduced_embedding_dimension=16, 
                    number_of_heads=8, 
                    max_length=dataset_config.max_len, 
                    forward_expansion=3, 
                    num_layers=2,
                    vocab_size=dataset_config.dict_size
                )

# wandb.login()


#load the dataset
print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}Importing dataset{RESET}")
train_dataset, test_dataset = dataset_selector(dataset_config.dataset_name)
dataset_config.labels = train_dataset.unique("label")
print(f"{FOREGROUND_COLORS["BrightCyan"]}Dataset contains {len(train_dataset)} training samples and {len(test_dataset)} test samples with {dataset_config.labels} labels{RESET}")


#load the tokenizer
print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}Loading/building tokenizer{RESET}")
print(f"{FOREGROUND_COLORS["BrightCyan"]}", end="")
tokenizer = make_tokenizer(dataset_config, train_dataset)
print(f"{RESET}")

#split the training to have a small validation set
validation_dataset = train_dataset.train_test_split(test_size=0.05)
train_dataset, validation_dataset = validation_dataset["train"], validation_dataset["test"]

# tokenizing the dataset
print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}Tokenizing dataset{RESET}")
train_dataloader = encode_dataset(tokenizer, train_dataset, dataset_config.max_len, dataset_config.batch_size)
validation_dataloader = encode_dataset(tokenizer, validation_dataset, dataset_config.max_len, dataset_config.batch_size)
test_dataloader = encode_dataset(tokenizer, test_dataset, dataset_config.max_len, dataset_config.batch_size)

train_dataloader.shuffle = True


# Initializing model
print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}Initializing model{RESET}")

# model = Brav(model_config)
# model = Nano_Mlp_structured(model_config)
# model = Bert_efficient(model_config)
model = Nano_Bert_Efficient(model_config)
# model = Gray_BERT_Efficient(model_config)
# model = Mlp_structured(model_config)
model = Mamba_model(model_config)
# model = MamBra_model(model_config)
# model = Embedder_model(model_config)

model_config.model_name = model.__class__.__name__

# cls = Classifier_rms(model, model_out_sz=model_config.embedding_dimension, labels_num=dataset_config.n_labels()).to(device)
# cls = Smart_classifier(model, model_out_sz=model_config.embedding_dimension, hidden_state=1, labels_num=dataset_config.n_labels()).to(device)
cls = Tester_classifier(model, model_out_sz=model_config.embedding_dimension, labels_num=dataset_config.n_labels()).to(device)

print(f"{FOREGROUND_COLORS["BrightCyan"]}", end="")
print_model_params(cls)
print(f"{RESET}")



# Training the model
print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}Training the model{RESET}")

trainer = Trainer(cls, device, lr, model_config)
trainer.train(train_dataloader, validation_dataloader, epochs, 3)

# Testing the model
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


# if not os.path.exists(f"results/{model_config.model_name}/"):
#     os.makedirs(f"results/{model_config.model_name}/")

# #save the classification report in a file for later use specifying the dataset, model hyperparameters
# with open(f"results/{model.__class__.__name__}/{dataset_config.dataset_name}_{dataset_config.dict_size}_{dataset_config.tokenizer_type}_cls_report.txt", "a") as f:
# 	f.write(f"{model_config}\n")
# 	f.write(f"LR: {lr}\n")
# 	f.write(f"EPOCHS: {epochs}\n\n")
# 	f.write(f"average eval loss: {avg_eval_loss: .4f}\n")
# 	f.write(f"{metrics_to_str(metrics)}\n")
# 	f.write(str(model_size(cls)))
# 	f.write("\n\n*******************************************\n\n")
