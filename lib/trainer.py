from colors import ATTRIBUTES, FOREGROUND_COLORS, RESET

from tqdm import tqdm
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, AdamW, Adam, SGD, RMSprop, Adagrad
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix

from lib.configs import ModelConfig, DataConfig
from lib.utils import checkpoint, resume, spearman_correlation



def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
			def lr_lambda(current_step):
				learning_rate = max(0.0, 1. - (float(current_step) / float(num_training_steps)))
				learning_rate *= min(1.0, float(current_step) / float(num_warmup_steps))
				return learning_rate
			return lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)




def mask_tokens(tokens, special_tokens_mask, dataset_config:DataConfig):


	#where tokens are masked
	aux = torch.count_nonzero(~special_tokens_mask.bool(), dim=1).tolist()
	aux = [(torch.randperm(aux[i]) + 1)[:max(1,aux[i]//6)] for i in range(tokens.shape[0])] #1/6 ~ 16% of the tokens are masked with at least 1 taken
	
	masked_indices = torch.zeros_like(tokens, device=tokens.device)
	for i in range(tokens.shape[0]):
		masked_indices[i, aux[i]] = 1
	
	#where inside the mask tokens are randomized
	random_indices = torch.bernoulli(masked_indices * 0.3).bool().to(tokens.device) # 30% of the 15% are 1 and 70% are 0 
	#where inside the mask tokens are replaced with themselves
	equal_indices = torch.bernoulli(random_indices * 0.5).bool().to(tokens.device) # 50% of the 30% are 1 and 50% are 0


	#making the three masks mutually exclusive
	masked_indices = (masked_indices * (~random_indices)).bool()
	random_indices = random_indices * (~equal_indices) 


	masked_tokens = tokens.clone()
	masked_tokens[masked_indices] = 3 # [MASK] token is 3
	masked_tokens[random_indices] = torch.randint(5, dataset_config.dict_size, masked_tokens[random_indices].shape, device=tokens.device) # random token in the dictionary
	masked_tokens[equal_indices] = tokens[equal_indices] # token is replaced with itself

	mask = masked_indices + random_indices + equal_indices

	return masked_tokens, mask.bool()


class Trainer:
	def __init__(self, model:nn.Module, device, model_config:ModelConfig):
		self.model = model
		#self.optimizer = AdamW(lr=model_config.learning_rate, params=self.model.parameters())
		self.optimizer = RMSprop(lr=model_config.learning_rate, params=self.model.parameters())
		self.device = device

		self.model_config = model_config
		self.savepath = f"trained_models/best_model_{model_config.model_name}_{model.__class__.__name__}.pth"
		
		self.regression = None


	def train(self, train_dataloader, eval_dataloader, num_epochs, log_freq: int, color = FOREGROUND_COLORS['Green']):

		log_step = len(train_dataloader) // log_freq

		#scaler = GradScaler()
		
		# scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=len(train_dataloader), num_training_steps=num_epochs*len(train_dataloader))
		# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=log_freq)

		

		# Extract labels from the data
		try:
			class_weights = torch.bincount(train_dataloader.dataset["label"])
			class_weights = class_weights.sum() / (class_weights * len(class_weights))
			print(f"Class weights: {class_weights}")
			self.criterion = nn.CrossEntropyLoss(weight=class_weights).to(self.device)
			self.regression = False

		except:
			self.criterion = nn.MSELoss().to(self.device)
			self.regression = True



		self.model.train()
		train_loss = 0.0

		max_val = -1


		for epoch in range(num_epochs):

			tqdm_train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
			for step_num, batch_data in enumerate(tqdm_train_loader):

				self.optimizer.zero_grad()

				tokens = batch_data["tokens"].to(self.device)
				masks  = batch_data["attention_mask"].to(self.device)
				labels = batch_data["label"].to(self.device)

				#Model outputs (batch_size, n_labels)
				y = self.model(tokens, masks).squeeze(-1)
				batch_loss = self.criterion(y, labels)

				if step_num == 0:
					train_loss = batch_loss.item()
				else:
					train_loss = 0.9 * train_loss + 0.1 * batch_loss.item()

				tqdm_train_loader.set_postfix(loss = "{:.4f}".format(train_loss))

				# Scales loss.  Calls backward() on scaled loss to create scaled gradients.
				# Backward passes under autocast are not recommended.
				# Backward ops run in the same dtype autocast chose for corresponding forward ops.
				# scaler.scale(batch_loss).backward()
				batch_loss.backward()

				# scaler.step() first unscales the gradients of the optimizer's assigned params.
				# If these gradients do not contain infs or NaNs, optimizer.step() is then called,
				# otherwise, optimizer.step() is skipped.
				# scaler.step(self.optimizer)
				self.optimizer.step()

				# scaler.update()

				# Update learning rate
				# scheduler.step()
				
				#print the accuracy on the training set
				# acc = accuracy_score(labels.cpu().detach(), torch.argmax(y, dim=1).cpu().detach())
				# tqdm.write(f"Step {step_num+1}/{len(train_dataloader)}:\t\t Loss: {batch_loss:.3f} Accuracy: {acc:.3f}")

				if step_num % log_step == (log_step - 1):
					self.model.eval()

					guesses = torch.Tensor([]).to(self.device)
					true_labels = torch.Tensor([]).to(self.device)
					val_loss = 0

					for batch_val in eval_dataloader:
						tokens = batch_val["tokens"].to(self.device)
						masks  = batch_val["attention_mask"].to(self.device)
						labels_val = batch_val["label"].to(self.device)

						with torch.no_grad():
							guess = self.model(tokens, masks).squeeze(-1)
						
						# print(f"labels: {labels_val}\nguess: {torch.argmax(guess, dim=1)}")

						val_loss += self.criterion(guess, labels_val).item()
						
						if not self.regression:
							guesses = torch.cat((guesses, torch.argmax(guess, dim=1)))
						else:
							guesses = torch.cat((guesses, guess))
						
						true_labels = torch.cat((true_labels, labels_val))

					guesses = guesses.cpu().detach()
					true_labels = true_labels.cpu().detach()

					if not self.regression:
						val_accuracy = accuracy_score(true_labels, guesses)
						mcc = matthews_corrcoef(true_labels, guesses)
						val_loss = val_loss / len(eval_dataloader)

						if(max_val < abs(mcc)):
							max_val = abs(mcc)
							checkpoint(self.model, self.savepath)
						
						tqdm.write(f"{RESET}Val loss: {val_loss:.3f}, Val accuracy: {val_accuracy:.3f}, Val mcc: {mcc:.3f} {color}")
						# tqdm.write(f"{RESET}Val loss: {val_loss:.3f}, Val accuracy: {val_accuracy:.3f}, Val mcc: {mcc:.3f}, Lr: {scheduler.get_last_lr()} {color}")
					else:
						val_loss = val_loss / len(eval_dataloader)
						spm_corr = spearman_correlation(true_labels, guesses)
						
						if(max_val < spm_corr):
							max_val = spm_corr
							checkpoint(self.model, self.savepath)

						tqdm.write(f"{RESET}Val loss: {val_loss:.3f} Spearman correlation {spm_corr:.3f} {color}")
						
					self.model.train()
		resume(self.model, self.savepath)

	def evaluate(self, test_dataloader):

		self.model.eval()
		predicted = torch.Tensor([]).to(self.device)

		tqdm_test_loader = tqdm(test_dataloader, desc=f"Evaluation: ", leave=False)

		avg_loss = 0
		with torch.no_grad():
			for step_num, batch_data in enumerate(tqdm_test_loader):

				tokens = batch_data["tokens"].to(self.device)
				masks  = batch_data["attention_mask"].to(self.device)
				labels = batch_data["label"].to(self.device)

				logits = self.model(tokens, masks).squeeze()
				loss = self.criterion(logits, labels)
				avg_loss += loss.item()

				if not self.regression:
					predicted = torch.cat((predicted, torch.argmax(logits, dim = 1)))
				else:
					predicted = torch.cat((predicted, logits))
				
		average_loss = avg_loss / len(test_dataloader)
		print(f"average loss: {average_loss}\n")

		return predicted.tolist(), average_loss;





class BertTrainer:

	def __init__(self, model:nn.Module, device, model_config: ModelConfig, dataset_config: DataConfig, mask_prob: float = 0.15):

		self.device = device

		self.model = model
		self.model_config = model_config


		self.optimizer = AdamW(lr=model_config.learning_rate , params=self.model.parameters())

		self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
		self.nsp_criterion = nn.CrossEntropyLoss().to(self.device)

		self.dataset_config = dataset_config

		self.mask_prob = mask_prob

	def train(self, train_dataloader, num_epochs, log_t_interval: int, color = FOREGROUND_COLORS['Green']):
		print(f"{color}")

		# scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=len(train_dataloader), num_training_steps=num_epochs*len(train_dataloader))
		# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50, T_mult=2, eta_min=5e-5)
		# wm_up_pct = 0.05
		# s1 = torch.optim.lr_scheduler.LinearLR(self.optimizer, 1e-2, 1, num_epochs*len(train_dataloader)*wm_up_pct)
		# s2 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, num_epochs*len(train_dataloader)*(1-wm_up_pct), 1e-5)
		# scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, [s1, s2], [int(num_epochs*len(train_dataloader)*wm_up_pct)])

		# scaler = GradScaler()

		self.model.train()
		train_loss = 0.0

		start_time = time.time()
		cls_acc = 0.5
		cls_mcc = 0.0
		
		min_loss = float('inf')
		ckpt_counter = 0

		with open(f"trained_models/logs/{self.model_config.model_name}_train_log.txt", "a") as f:
			f.write(f"\nNEW TRAINING\n")
			f.write(f"Model: {self.model_config.model_name}\n")
			f.write(f"Time: {time.asctime(time.localtime(time.time()))}\n")

		for epoch in range(num_epochs):

			tqdm_train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
			for step_num, batch_data in enumerate(tqdm_train_loader):

				self.model.zero_grad()

				# "tokens", "attention_mask", "type_ids", "special_tokens_mask", "label"
				tokens = batch_data["tokens"].to(self.device)
				masks  = batch_data["attention_mask"].to(self.device)
				special_tokens_mask = batch_data["special_tokens_mask"].to(self.device)
				nsp_label = batch_data["label"].to(self.device)

				masked_tokens, masks_mask = mask_tokens(tokens, special_tokens_mask, self.dataset_config)

				with autocast():
					#Model outputs (batch_size, seq_len, dict_size) and (batch_size, seq_len, 2)
					logits, label_guess = self.model(masked_tokens, masks)

					tokens[~masks_mask] = 0

					lm_loss = self.mlm_criterion(logits.transpose(1,2), tokens)
					cls_loss = self.nsp_criterion(label_guess.squeeze(-1), nsp_label)

					batch_loss = lm_loss + cls_loss


				if step_num == 0:
					train_loss = batch_loss.item()
				else:
					train_loss = 0.9 * train_loss + 0.1 * batch_loss.item()

				# tqdm_train_loader.set_postfix(loss = "{:.4f}".format(train_loss), lr = f"{scheduler.get_last_lr()[0]:.6f}")
				tqdm_train_loader.set_postfix(loss = "{:.4f}".format(train_loss))

				# Scales loss.  Calls backward() on scaled loss to create scaled gradients.
				# Backward passes under autocast are not recommended.
				# Backward ops run in the same dtype autocast chose for corresponding forward ops.
				#scaler.scale(batch_loss).backward()
				batch_loss.backward()

				# scaler.step() first unscales the gradients of the optimizer's assigned params.
				# If these gradients do not contain infs or NaNs, optimizer.step() is then called,
				# otherwise, optimizer.step() is skipped.
				# scaler.step(self.optimizer)
				self.optimizer.step()


				#scaler.update()


				# Update learning rate
				# scheduler.step()


				if time.time() - start_time >= log_t_interval:
					#calculate the metrics
					cls_acc = cls_acc * 0.7 + 0.3 * torch.mean((torch.argmax(label_guess, dim=1) == nsp_label).float()).item()
					cls_mcc = cls_mcc * 0.7 + 0.3 * matthews_corrcoef(nsp_label.cpu().detach(), torch.argmax(label_guess, dim=1).cpu().detach())
					tqdm.write(f"Step {step_num+1}/{len(train_dataloader)}:\t\t MLM Loss: {lm_loss:.3f} CLS Loss: {cls_loss:.3f} CLS Accuracy: {cls_acc:.3f} CLS MCC: {cls_mcc:.3f}")
					
					#log the statistics in a log file
					with open(f"trained_models/logs/{self.model_config.model_name}_train_log.txt", "a") as f:
						f.write(f"Step {step_num+1}/{len(train_dataloader)}:\t\t MLM Loss: {lm_loss:.3f} CLS Loss: {cls_loss:.3f} CLS Accuracy: {cls_acc:.3f} CLS MCC: {cls_mcc:.3f}\n")

					#save the model if the loss is the minimum
					if(min_loss > batch_loss.item()):
						min_loss = batch_loss.item()
						#save with model name and ckpt counter
						checkpoint(self.model.model, f"trained_models/checkpoints/{self.model_config.model_name}_{ckpt_counter}.pth")
						ckpt_counter += 1
					
					# Reset the start time
					start_time = time.time()


				
				# if step_num % log_step == (log_step - 1):
				# 	self.model.eval()
				# 	mlm_accuracy = torch.Tensor([]).to(self.device)
				# 	cls_accuracy = torch.Tensor([]).to(self.device)
				# 	mlm_avg_loss = 0
				# 	cls_avg_loss = 0
				# 	val_loss = 0

				# 	for batch_val in eval_dataloader:
				# 		tokens = batch_val["tokens"].to(self.device)
				# 		masks  = batch_val["attention_mask"].to(self.device)
				# 		lables_val = batch_val["label"].to(self.device)

				# 		masked_tokens, masks_mask = mask_tokens(tokens, self.dataset_config)

				# 		logits, label_guess = self.model(masked_tokens, masks)

				# 		tokens[~masks_mask] = 0 #set to 0 the tokens that are not masked so that they are not considered in the loss
				# 		label_guess = label_guess.squeeze(-1) #remove the last dimension for the loss

				# 		lm_loss = self.lm_criterion(logits.transpose(1,2), tokens)
				# 		cls_loss = cls_criterion(label_guess, lables_val)

				# 		val_loss += lm_loss.item() + cls_loss.item()
				# 		mlm_avg_loss += lm_loss.item()
				# 		cls_avg_loss += cls_loss.item()

				# 		mlm_accuracy = torch.cat((mlm_accuracy, (torch.argmax(logits, dim=2)[masks_mask] == tokens[masks_mask])))
				# 		cls_accuracy = torch.cat((cls_accuracy, (torch.argmax(label_guess, dim=1) == lables_val)))


				# 	mlm_accuracy = torch.mean(mlm_accuracy.float()).item()
				# 	cls_accuracy = torch.mean(cls_accuracy.float()).item()
				# 	val_loss = val_loss / len(eval_dataloader)
				# 	cls_avg_loss = cls_avg_loss / len(eval_dataloader)
				# 	mlm_avg_loss = mlm_avg_loss / len(eval_dataloader)

				# 	#tqdm.write(f"{RESET}Val loss: {val_loss:.3f}, mlm loss: {mlm_avg_loss:.3f}, mlm accuracy: {mlm_accuracy:.3f}, cls loss: {cls_avg_loss:.3f}, cls accuracy: {cls_accuracy:.3f}, Lr: {scheduler.get_last_lr()} {color}")
				# 	tqdm.write(f"{RESET}Val loss: {val_loss:.3f}, mlm loss: {mlm_avg_loss:.3f}, mlm accuracy: {mlm_accuracy:.3f}, cls loss: {cls_avg_loss:.3f}, cls accuracy: {cls_accuracy:.3f}{color}")
				# 	self.model.train()
