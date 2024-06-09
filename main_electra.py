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
from lib.Models.final_classifiers import Classifier_rms, Classifier_BERT, Classifier_for_electra, Classifier_post_electra
from lib.Models.models import *


from lib.electra import Electra


epochs_pretraining = 30
lr_pretraining = 5e-3

epochs_post = 10
lr_post = 1e-3
logs_x_epoch = 2

dataset_config = DataConfig(
                    dataset_name="bull", 
                    dict_size=pow(2, 12), 
                    tokenizer_type="bpe", 
                    batch_size=128, 
                    max_len=64, 
                    labels=None
                )

generator_config = ModelConfig(
                    model_name=None, 
                    embedding_dimension=128, 
                    reduced_embedding_dimension=16, 
                    number_of_heads=8, 
                    max_length=dataset_config.max_len, 
                    forward_expansion=2, 
                    num_layers=1,
                    vocab_size=dataset_config.dict_size
                )

discriminator_config = ModelConfig(
					model_name=None, 
                    embedding_dimension=generator_config.embedding_dimension, 
                    reduced_embedding_dimension=generator_config.reduced_embedding_dimension, 
                    number_of_heads=8, 
                    max_length=dataset_config.max_len, 
                    forward_expansion=4, 
                    num_layers=2,
                    vocab_size=dataset_config.dict_size
                )

########################################################################################
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
validation_dataset = train_dataset.train_test_split(test_size=0.1)
train_dataset, validation_dataset = validation_dataset["train"], validation_dataset["test"]

# tokenizing the dataset
print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}Tokenizing dataset{RESET}")
train_dataloader = encode_dataset(tokenizer, train_dataset, dataset_config.max_len, dataset_config.batch_size)
validation_dataloader = encode_dataset(tokenizer, validation_dataset, dataset_config.max_len, dataset_config.batch_size)
test_dataloader = encode_dataset(tokenizer, test_dataset, dataset_config.max_len, dataset_config.batch_size)

train_dataloader.shuffle = True

########################################################################################
print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS["BrightYellow"]}Initializing model{RESET}")

# config = MambaConfig(d_model=EMBED_DIM, n_layers=LAYERS, expand_factor=FORWARD_EXPANSION)
# model = Mamba(config)
# cls = Mamba_classifier(model, EMBED_DIM, REDUCED_EMBEDDING_DIM, VOCAB_SIZE, N_LABELS)


#choose generator model
# generator = Brav(generator_config)
# generator = Bert_efficient(generator_config)
generator = Nano_Bert_Efficient(generator_config)
# generator = Mlp_structured(generator_config)

#choose discriminator model
# discriminator = Brav(discriminator_config)
# discriminator = Bert_efficient(discriminator_config)
discriminator = Nano_Bert_Efficient(discriminator_config)
# discriminator = Mlp_structured(discriminator_config)


generator.embedder.token.weight.data = discriminator.embedder.token.weight.data
generator.embedder.position.weight.data = discriminator.embedder.position.weight.data

generator_with_classifier = Classifier_for_electra(generator, generator_config.embedding_dimension, dataset_config.dict_size)
discriminator_with_classifier = Classifier_for_electra(discriminator, discriminator_config.embedding_dimension, 1)


electra = Electra(
    generator_with_classifier,
    discriminator_with_classifier,
    mask_token_id = tokenizer.token_to_id("[MASK]"),          	# the token id reserved for masking
    pad_token_id = tokenizer.token_to_id("[PAD]"),				# the token id for padding
    mask_prob = 0.15,           								# masking probability for masked language modeling
    mask_ignore_token_ids = [0,1,2,3,4]  						# ids of tokens to ignore for mask modeling ex. (cls, sep)
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
electra.to(device)
print(f"Model {electra.__class__.__name__} initialized on {device}")


print("Generator parameters:")
#print all model parameters with names
print_model_params(generator)

print("Discriminator parameters:")
#print all model parameters with names
print_model_params(discriminator)


########################################################################################
print(f"{ATTRIBUTES['Bold']}Starting training{RESET}")

	

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        learning_rate = max(0.0, 1. - (float(current_step) / float(num_training_steps)))
        learning_rate *= min(1.0, float(current_step) / float(num_warmup_steps))
        return learning_rate
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def get_params_without_weight_decay_ln(named_params, weight_decay):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in named_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        return optimizer_grouped_parameters

optimizer = torch.optim.AdamW(get_params_without_weight_decay_ln(electra.named_parameters(), weight_decay=0.1), lr=lr_pretraining)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader), num_training_steps=epochs_pretraining*len(train_dataloader)*2)


log_step = len(train_dataloader) // logs_x_epoch
electra.train()
for epoch in range(epochs_pretraining):
	tqdm.write(f"{FOREGROUND_COLORS['Green']}Epoch {epoch+1}/{epochs_pretraining}")
	
	train_loss = 0
	tqdm_train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs_pretraining}", leave=False)

	for step_num, batch_data in enumerate(tqdm_train_loader):

		tokens = batch_data["tokens"].to(device) 
		masks  = batch_data["attention_mask"].to(device)
		labels = batch_data["label"].to(device)

		optimizer.zero_grad()

		with torch.cuda.amp.autocast():
			loss, loss_mlm, loss_disc, acc_gen, acc_disc, disc_labels, disc_pred = electra(tokens, mask = masks)
		
		loss.backward()
		optimizer.step()
		scheduler.step()	

		metrics = {
            'loss': (loss.item(), '{:8.5f}'),
            'loss_mlm': (loss_mlm.item(), '{:8.5f}'),
            'loss_disc': (loss_disc.item(), '{:8.5f}'),
            'acc_gen': (acc_gen.item(), '{:5.3f}'),
            'acc_disc': (acc_disc.item(), '{:5.3f}'),
            'lr': (scheduler.get_last_lr()[0], '{:8.7f}'),
        }

		if step_num == 0:
			train_loss = loss.item()
		else:
			train_loss = 0.99 * train_loss + 0.01 * loss.item()



		loss_str = "{:.4f}".format(train_loss)
		tqdm_train_loader.set_postfix(loss = loss_str)			
		if step_num % log_step == (log_step - 1):
			electra.eval()
			val_gen_accuracy = 0
			val_disc_accuracy = 0
			val_loss = 0
			for batch_val in validation_dataloader:
				tokens = batch_val["tokens"].to(device) 
				masks  = batch_val["attention_mask"].to(device)
				labels_val = batch_val["label"].to(device)
				
				loss, _, _, acc_gen, acc_disc, _, _ = electra(tokens, mask = masks)
				val_loss += loss.item()
				val_gen_accuracy += acc_gen.item()
				val_disc_accuracy += acc_disc.item()
				
			val_gen_accuracy = val_gen_accuracy / len(validation_dataloader)
			val_disc_accuracy = val_disc_accuracy / len(validation_dataloader)
			val_loss = val_loss / len(validation_dataloader)

			#mcc = matthews_corrcoef(y.cpu().numpy(), torch.argmax(guess, dim=1).cpu().numpy())
			tqdm.write(f"{RESET}Val loss: {val_loss:.3f}, Val gen acc: {val_gen_accuracy:.3f}, Val disc acc: {val_disc_accuracy:.3f}, Lr: {scheduler.get_last_lr()[0]:.6f}{FOREGROUND_COLORS['Green']}")
			electra.train()
			
print(f"{RESET}")	

# ########################################################################################
# print(f"{ATTRIBUTES['Bold']}Starting evaluation{RESET}")
# predicted, avg_eval_loss = utils.evaluator(cls, test_dataloader)

# accuracy = accuracy_score(label_test, predicted)
# f1 = f1_score(label_test, predicted, average='micro')  
# precision = precision_score(label_test, predicted, average='micro')  
# recall = recall_score(label_test, predicted, average='micro')
# mcc = matthews_corrcoef(label_test, predicted)
# conf_mat = confusion_matrix(label_test, predicted)

# print(f"Accuracy: {accuracy}")
# print(f"F1: {f1}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"MCC: {mcc}")
# print(f"Confusion matrix:\n {conf_mat}")





########################################################################################
print(f"{ATTRIBUTES['Bold']}Normal model initialization:{RESET}")

classifier = Classifier_BERT(discriminator, discriminator_config.embedding_dimension, dataset_config.n_labels())
classifier.to(device)
print(f"Model initialized on {device}")


print("Model parameters:")
#print all model parameters with names
print_model_params(classifier)

########################################################################################

print(f"{ATTRIBUTES['Bold']}Starting training{RESET}")

epochs_post = 5
lr_post = 1e-4

trainer(classifier, train_dataloader, validation_dataloader, lr=lr_post, epochs=epochs_post, logs_x_epoch=logs_x_epoch)


# ########################################################################################
# print(f"{ATTRIBUTES['Bold']}Starting evaluation{RESET}")
# predicted, avg_eval_loss = utils.evaluator(std_cls, test_dataloader)

# accuracy = accuracy_score(label_test, predicted)
# f1 = f1_score(label_test, predicted, average='micro')  
# precision = precision_score(label_test, predicted, average='micro')  
# recall = recall_score(label_test, predicted, average='micro')
# mcc = matthews_corrcoef(label_test, predicted)
# conf_mat = confusion_matrix(label_test, predicted)


# print(f"Accuracy: {accuracy}")
# print(f"F1: {f1}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"MCC: {mcc}")
# print(f"Confusion matrix:\n {conf_mat}")



# ########################################################################################
# #save the classification report in a file for later use specifying the dataset, model hyperparameters
# with open(f"results/tests_{DATASET}_classification_report.txt", "a") as f:
# 	f.write(f"Values of final training\n")
# 	f.write(f"MODEL: {model.__class__.__name__}\n")
# 	f.write(f"DATASET: {DATASET}\n")
# 	f.write(f"VOCAB_SIZE: {VOCAB_SIZE}\n")
# 	f.write(f"EMBED_DIM: {EMBED_DIM}\n")
# 	f.write(f"FORWARD_EXPANSION: {FORWARD_EXPANSION}\n")
# 	f.write(f"MAX_LENGTH: {MAX_LENGTH}\n")
# 	f.write(f"LAYERS: {LAYERS}\n")
# 	f.write(f"REDUCED_EMBEDDING_DIM: {REDUCED_EMBEDDING_DIM}\n")
# 	f.write(f"LR: {LR}\n")
# 	f.write(f"EPOCHS: {EPOCHS}\n\n")
# 	f.write(f"average eval loss: {avg_eval_loss}\n")
# 	f.write(f"Accuracy: {accuracy}\n")
# 	f.write(f"F1 (weighted): {f1}\n")
# 	f.write(f"Precision (weighted): {precision}\n")
# 	f.write(f"Recall (weighted): {recall}\n")
# 	f.write(f"MCC: {mcc}\n")
# 	f.write(f"Confusion matrix:\n {conf_mat}\n")
# 	f.write(str(utils.model_size(std_cls)))
# 	f.write("\n\n*******************************************\n\n")

