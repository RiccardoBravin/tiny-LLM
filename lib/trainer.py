from colors import ATTRIBUTES, FOREGROUND_COLORS, RESET

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, AdamW, Adam
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix

from lib.configs import ModelConfig, DataConfig


def checkpoint(model, filename):
	torch.save(model.state_dict(), filename)

def resume(model, filename):
	model.load_state_dict(torch.load(filename))

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
			def lr_lambda(current_step):
				learning_rate = max(0.0, 1. - (float(current_step) / float(num_training_steps)))
				learning_rate *= min(1.0, float(current_step) / float(num_warmup_steps))
				return learning_rate
			return lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)




def mask_tokens(tokens, dataset_config:DataConfig):

	#where tokens are masked
	mask = torch.ones_like(tokens, device=tokens.device) * (tokens > len(dataset_config.special_tokens))
	
	aux = torch.count_nonzero(mask, dim=1).tolist()
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
		self.optimizer = AdamW(lr=model_config.learning_rate, params=self.model.parameters(), amsgrad=True)
		self.device = device

		self.model_config = model_config
		self.savepath = f"trained_models/best_model_{model_config.model_name}_{model.__class__.__name__}.pth"


	def train(self, train_dataloader, eval_dataloader, num_epochs, log_freq: int, color = FOREGROUND_COLORS['Green']):

		log_step = len(train_dataloader) // log_freq

		#scaler = GradScaler()


		# Extract labels from the data
		class_weights = torch.bincount(train_dataloader.dataset["label"])
		class_weights = class_weights.sum() / class_weights

		# scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=len(train_dataloader), num_training_steps=num_epochs*len(train_dataloader))
		# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=log_freq)
		self.criterion = nn.CrossEntropyLoss(weight=class_weights).to(self.device)


		self.model.train()
		train_loss = 0.0

		max_mcc = -1


		for epoch in range(num_epochs):

			tqdm_train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
			for step_num, batch_data in enumerate(tqdm_train_loader):

				self.model.zero_grad()

				tokens = batch_data["tokens"].to(self.device)
				masks  = batch_data["attention_mask"].to(self.device)
				labels = batch_data["label"].to(self.device)

				#Model outputs (batch_size, n_labels)
				y = self.model(tokens, masks)
				batch_loss = self.criterion(y, labels)


				if step_num == 0:
					train_loss = batch_loss.item()
				else:
					train_loss = 0.9 * train_loss + 0.1 * batch_loss.item()

				tqdm_train_loader.set_postfix(loss = "{:.4f}".format(train_loss))

				# Scales loss.  Calls backward() on scaled loss to create scaled gradients.
				# Backward passes under autocast are not recommended.
				# Backward ops run in the same dtype autocast chose for corresponding forward ops.
				#scaler.scale(batch_loss).backward()
				batch_loss.backward()

				# scaler.step() first unscales the gradients of the optimizer's assigned params.
				# If these gradients do not contain infs or NaNs, optimizer.step() is then called,
				# otherwise, optimizer.step() is skipped.
				#scaler.step(self.optimizer)
				self.optimizer.step()


				# scaler.update()
				# Update learning rate
				# scheduler.step()

				if step_num % log_step == (log_step - 1):
					self.model.eval()

					guesses = torch.Tensor([]).to(self.device)
					val_loss = 0

					for batch_val in eval_dataloader:
						tokens = batch_val["tokens"].to(self.device)
						masks  = batch_val["attention_mask"].to(self.device)
						labels_val = batch_val["label"].to(self.device)

						guess = self.model(tokens, masks)

						val_loss += self.criterion(guess, labels_val).item()

						guesses = torch.cat((guesses, torch.argmax(guess, dim=1)))

					guesses = guesses.cpu().detach()
					val_accuracy = accuracy_score(eval_dataloader.dataset["label"], guesses)
					mcc = matthews_corrcoef(eval_dataloader.dataset["label"], guesses)
					val_loss = val_loss / len(eval_dataloader)

					# scheduler.step(-mcc)
					if(max_mcc < mcc):
						max_mcc = mcc
						checkpoint(self.model, self.savepath)

					# tqdm.write(f"{RESET}Val loss: {val_loss:.3f}, Val accuracy: {val_accuracy:.3f}, Val mcc: {mcc:.3f}, Lr: {scheduler.get_last_lr()} {color}")
					tqdm.write(f"{RESET}Val loss: {val_loss:.3f}, Val accuracy: {val_accuracy:.3f}, Val mcc: {mcc:.3f} {color}")
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

				logits = self.model(tokens, masks)
				loss = self.criterion(logits, labels)
				avg_loss += loss.item()

				predicted = torch.cat((predicted, torch.argmax(logits, dim = 1)))

		average_loss = avg_loss / len(test_dataloader)
		print(f"average loss: {average_loss}\n")

		return predicted.tolist(), average_loss;





class BertTrainer:

	def __init__(self, model:nn.Module, device, model_config: ModelConfig, dataset_config: DataConfig, mask_prob: float = 0.15):

		self.device = device

		self.model = model
		self.model_config = model_config


		self.optimizer = AdamW(lr=model_config.pretraining_lr , params=self.model.parameters())

		self.lm_criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)


		self.dataset_config = dataset_config

		self.mask_prob = mask_prob

	def train(self, train_dataloader, eval_dataloader, num_epochs, log_freq: int, color = FOREGROUND_COLORS['Green']):
		print(f"{color}")

		log_step = len(train_dataloader) // log_freq

		#scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=len(train_dataloader), num_training_steps=num_epochs*len(train_dataloader))

		# Extract labels from the data
		class_weights = torch.bincount(train_dataloader.dataset["label"])
		class_weights = class_weights.sum() / class_weights

		# scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=len(train_dataloader), num_training_steps=num_epochs*len(train_dataloader))
		cls_criterion = nn.CrossEntropyLoss(weight=class_weights).to(self.device)


		self.model.train()
		train_loss = 0.0

		for epoch in range(num_epochs):

			tqdm_train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
			for step_num, batch_data in enumerate(tqdm_train_loader):

				self.model.zero_grad()

				tokens = batch_data["tokens"].to(self.device)
				masks  = batch_data["attention_mask"].to(self.device)
				lables = batch_data["label"].to(self.device)

				masked_tokens, masks_mask = mask_tokens(tokens, self.dataset_config)


				with autocast():
					#Model outputs (batch_size, seq_len, dict_size) and (batch_size, seq_len, 2)
					logits, label_guess = self.model(masked_tokens, masks)

					tokens[~masks_mask] = 0

					lm_loss = self.lm_criterion(logits.transpose(1,2), tokens)
					cls_loss = cls_criterion(label_guess.squeeze(-1), lables)

					batch_loss = lm_loss + cls_loss


				if step_num == 0:
					train_loss = batch_loss.item()
				else:
					train_loss = 0.9 * train_loss + 0.1 * batch_loss.item()

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
				#scheduler.step()


				if step_num % log_step == (log_step - 1):
					self.model.eval()
					mlm_accuracy = torch.Tensor([]).to(self.device)
					cls_accuracy = torch.Tensor([]).to(self.device)
					mlm_avg_loss = 0
					cls_avg_loss = 0
					val_loss = 0

					for batch_val in eval_dataloader:
						tokens = batch_val["tokens"].to(self.device)
						masks  = batch_val["attention_mask"].to(self.device)
						lables_val = batch_val["label"].to(self.device)

						masked_tokens, masks_mask = mask_tokens(tokens, self.dataset_config)

						logits, label_guess = self.model(masked_tokens, masks)

						tokens[~masks_mask] = 0 #set to 0 the tokens that are not masked so that they are not considered in the loss
						label_guess = label_guess.squeeze(-1) #remove the last dimension for the loss

						lm_loss = self.lm_criterion(logits.transpose(1,2), tokens)
						cls_loss = cls_criterion(label_guess, lables_val)

						val_loss += lm_loss.item() + cls_loss.item()
						mlm_avg_loss += lm_loss.item()
						cls_avg_loss += cls_loss.item()

						mlm_accuracy = torch.cat((mlm_accuracy, (torch.argmax(logits, dim=2)[masks_mask] == tokens[masks_mask])))
						cls_accuracy = torch.cat((cls_accuracy, (torch.argmax(label_guess, dim=1) == lables_val)))


					mlm_accuracy = torch.mean(mlm_accuracy.float()).item()
					cls_accuracy = torch.mean(cls_accuracy.float()).item()
					val_loss = val_loss / len(eval_dataloader)
					cls_avg_loss = cls_avg_loss / len(eval_dataloader)
					mlm_avg_loss = mlm_avg_loss / len(eval_dataloader)

					#tqdm.write(f"{RESET}Val loss: {val_loss:.3f}, mlm loss: {mlm_avg_loss:.3f}, mlm accuracy: {mlm_accuracy:.3f}, cls loss: {cls_avg_loss:.3f}, cls accuracy: {cls_accuracy:.3f}, Lr: {scheduler.get_last_lr()} {color}")
					tqdm.write(f"{RESET}Val loss: {val_loss:.3f}, mlm loss: {mlm_avg_loss:.3f}, mlm accuracy: {mlm_accuracy:.3f}, cls loss: {cls_avg_loss:.3f}, cls accuracy: {cls_accuracy:.3f}{color}")
					self.model.train()
