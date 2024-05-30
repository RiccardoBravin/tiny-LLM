import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix

from colors import ATTRIBUTES, FOREGROUND_COLORS, RESET

from lib import utils
from lib.MAMBA import Mamba, MambaConfig, Mamba_classifier
from lib.BERT_Eff import BERT_efficient
from lib.BERT_Eff_Enc_Head import BERT_Eff_Multihead
from lib.BRAV_2 import BRAV_2
from lib.MLP import MLPSwiGLU

from lib.electra import Electra


from lib.classifier import classifier


from lib.dataset import dataset_importer

VOCAB_SIZE = 512*8
BATCH_SIZE = 16

REDUCED_EMBEDDING_DIM = 16
EMBED_DIM = 32
#EMBED_DIM = 16
NUM_HEADS = 8
FORWARD_EXPANSION = 8
MAX_LENGTH = 128
LAYERS = 2

########################################################################################
print(f"{ATTRIBUTES['Bold']}Loading dataset:{RESET}")

#'Amazon', "imdb", "sst2" "twitter" "race" "yelp" "news" "trec_coarse" "bull" "limit" "dbpedia" "nlu" "snips" "blog"
DATASET = "bull"

train_dataloader, val_dataloader, test_dataloader, LABELS, label_test = dataset_importer(DATASET, VOCAB_SIZE, MAX_LENGTH, BATCH_SIZE)
N_LABELS = len(LABELS)

########################################################################################
print(f"{ATTRIBUTES['Bold']}Model initialization:{RESET}")

# config = MambaConfig(d_model=EMBED_DIM, n_layers=LAYERS, expand_factor=FORWARD_EXPANSION)
# model = Mamba(config)
# cls = Mamba_classifier(model, EMBED_DIM, REDUCED_EMBEDDING_DIM, VOCAB_SIZE, N_LABELS)

#generator = MLPSwiGLU(VOCAB_SIZE, EMBED_DIM, REDUCED_EMBEDDING_DIM, LAYERS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1) 

# model = MLPSwiGLU(VOCAB_SIZE, EMBED_DIM, LAYERS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)
# model = BERT_Efficient(VOCAB_SIZE, EMBED_DIM, LAYERS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)
# model = BERT_Eff_gray(VOCAB_SIZE, EMBED_DIM, REDUCED_EMBEDDING_DIM, LAYERS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)
# model = BERT_Eff_multihead(VOCAB_SIZE, EMBED_DIM, LAYERS, NUM_HEADS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)	
# model = BRAV(VOCAB_SIZE, EMBED_DIM, LAYERS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)
#model = BRAV_multihead(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, LAYERS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)
model = BRAV_2(VOCAB_SIZE, EMBED_DIM, REDUCED_EMBEDDING_DIM, LAYERS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)
#model = Gated_BERT(VOCAB_SIZE, EMBED_DIM, LAYERS, NUM_HEADS, MAX_LENGTH, FORWARD_EXPANSION, dropout=0.1)
generator = BRAV_2(VOCAB_SIZE, EMBED_DIM, REDUCED_EMBEDDING_DIM, 1, MAX_LENGTH, 4, dropout=0.1)



generator.embedder.token.weight.data = model.embedder.token.weight.data
generator.embedder.position.weight.data = model.embedder.position.weight.data

generator_with_adapter = torch.nn.Sequential(generator, torch.nn.Linear(EMBED_DIM, VOCAB_SIZE))
discriminator_with_adapter = torch.nn.Sequential(model, torch.nn.Linear(EMBED_DIM, 1))

electra = Electra(
    generator_with_adapter,
    discriminator_with_adapter,
    mask_token_id = 4,          # the token id reserved for masking
    pad_token_id = 0,           # the token id for padding
    mask_prob = 0.15,           # masking probability for masked language modeling
    mask_ignore_token_ids = [1,2,3]  # ids of tokens to ignore for mask modeling ex. (cls, sep)
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
electra.to(device)
print(f"Model {electra.__class__.__name__} initialized on {device}")


print("Generator parameters:")
#print all model parameters with names
for name, param in generator.named_parameters():
	print(f"{name}: {param.nelement()}")

#print the model size
print(utils.model_size(generator))


print("Model parameters:")
#print all model parameters with names
for name, param in model.named_parameters():
	print(f"{name}: {param.nelement()}")
	
#print the model size
print(utils.model_size(model))


########################################################################################
print(f"{ATTRIBUTES['Bold']}Starting training{RESET}")

EPOCHS = 10
LR = 1e-2
	

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

optimizer = torch.optim.AdamW(get_params_without_weight_decay_ln(electra.named_parameters(), weight_decay=0.1), lr=LR)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)/2, num_training_steps=EPOCHS*len(train_dataloader)*2)


log_step = len(train_dataloader) // 10
electra.train()
for epoch in range(EPOCHS):
	tqdm.write(f"{FOREGROUND_COLORS['Green']}Epoch {epoch+1}/{EPOCHS}")
	
	train_loss = 0
	tqdm_train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

	for step_num, batch_data in enumerate(tqdm_train_loader):

		token_ids, _ = tuple(t.to(device) for t in batch_data)

		optimizer.zero_grad()

		with torch.cuda.amp.autocast():
			loss, loss_mlm, loss_disc, acc_gen, acc_disc, disc_labels, disc_pred = electra(token_ids)
		
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
			model.eval()
			val_gen_accuracy = 0
			val_disc_accuracy = 0
			val_loss = 0
			for x, y in val_dataloader:
				x, y = x.to(device), y.to(device)

				loss, _, _, acc_gen, acc_disc, _, _ = electra(x)
				val_loss += loss.item()
				val_gen_accuracy += acc_gen.item()
				val_disc_accuracy += acc_disc.item()
				
			val_gen_accuracy = val_gen_accuracy / len(val_dataloader)
			val_disc_accuracy = val_disc_accuracy / len(val_dataloader)
			val_loss = val_loss / len(val_dataloader)

			#mcc = matthews_corrcoef(y.cpu().numpy(), torch.argmax(guess, dim=1).cpu().numpy())
			tqdm.write(f"{RESET}Val loss: {val_loss:.3f}, Val gen acc: {val_gen_accuracy:.3f}, Val disc acc: {val_disc_accuracy:.3f}, Lr: {scheduler.get_last_lr()[0]:.6f}{FOREGROUND_COLORS['Green']}")
			model.train()
			
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

std_cls = classifier(model, EMBED_DIM, N_LABELS)
std_cls.to(device)
print(f"Model initialized on {device}")


print("Model parameters:")
#print all model parameters with names
for name, param in std_cls.named_parameters():
	print(f"{name}: {param.nelement()}")

#print the model size
print(utils.model_size(std_cls))

########################################################################################

print(f"{ATTRIBUTES['Bold']}Starting training{RESET}")

EPOCHS = 5
LR = 1e-4

utils.trainer(std_cls, train_dataloader, val_dataloader, LR, EPOCHS)


########################################################################################
print(f"{ATTRIBUTES['Bold']}Starting evaluation{RESET}")
predicted, avg_eval_loss = utils.evaluator(std_cls, test_dataloader)

accuracy = accuracy_score(label_test, predicted)
f1 = f1_score(label_test, predicted, average='micro')  
precision = precision_score(label_test, predicted, average='micro')  
recall = recall_score(label_test, predicted, average='micro')
mcc = matthews_corrcoef(label_test, predicted)
conf_mat = confusion_matrix(label_test, predicted)


print(f"Accuracy: {accuracy}")
print(f"F1: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"MCC: {mcc}")
print(f"Confusion matrix:\n {conf_mat}")



########################################################################################
#save the classification report in a file for later use specifying the dataset, model hyperparameters
with open(f"results/tests_{DATASET}_classification_report.txt", "a") as f:
	f.write(f"Values of final training\n")
	f.write(f"MODEL: {model.__class__.__name__}\n")
	f.write(f"DATASET: {DATASET}\n")
	f.write(f"VOCAB_SIZE: {VOCAB_SIZE}\n")
	f.write(f"EMBED_DIM: {EMBED_DIM}\n")
	f.write(f"FORWARD_EXPANSION: {FORWARD_EXPANSION}\n")
	f.write(f"MAX_LENGTH: {MAX_LENGTH}\n")
	f.write(f"LAYERS: {LAYERS}\n")
	f.write(f"REDUCED_EMBEDDING_DIM: {REDUCED_EMBEDDING_DIM}\n")
	f.write(f"LR: {LR}\n")
	f.write(f"EPOCHS: {EPOCHS}\n\n")
	f.write(f"average eval loss: {avg_eval_loss}\n")
	f.write(f"Accuracy: {accuracy}\n")
	f.write(f"F1 (weighted): {f1}\n")
	f.write(f"Precision (weighted): {precision}\n")
	f.write(f"Recall (weighted): {recall}\n")
	f.write(f"MCC: {mcc}\n")
	f.write(f"Confusion matrix:\n {conf_mat}\n")
	f.write(str(utils.model_size(std_cls)))
	f.write("\n\n*******************************************\n\n")

