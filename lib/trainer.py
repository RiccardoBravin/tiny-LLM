from colors import ATTRIBUTES, FOREGROUND_COLORS, RESET

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, AdamW
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix

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


def mask_tokens(tokens, dict_size, device, mlm_prob = 0.15):
	
	mlm_probability = torch.full(tokens.shape, mlm_prob).to(tokens.device)

	mlm_probability = mlm_probability * (tokens > 4) #set to 0 the probability of masking the special tokens
	
	
	masked_indices = torch.bernoulli(mlm_probability).bool() # 15% of the tokens are 1 and 85% are 0
	randomized_indices = torch.bernoulli(masked_indices * 0.4).bool().to(device) # 40% of the 15% are 1 and 60% are 0
	
	masked_tokens = tokens.clone()
	masked_tokens[masked_indices * (~randomized_indices)] = 3 # [MASK] token is 3
	masked_tokens[masked_indices * randomized_indices] = torch.randint(5, dict_size, masked_tokens[masked_indices * randomized_indices].shape).to(device) # random token in the dictionary
	
	return masked_tokens, masked_indices


def split_and_randomize_phrases(tokens, masks):
	#take the first half of the tokens and masks and shuffle them
	half_batch = tokens.shape[0] // 2
	counts = min([torch.count_nonzero(masks[i]) for i in range(half_batch)])//2 #could be redone since it is efficient but not so random

	phrase_a = tokens[:half_batch, :counts]
	phrase_b = tokens[:half_batch, counts:-1]
	phrase_a_mask = masks[:half_batch, :counts]
	phrase_b_mask = masks[:half_batch, counts:]


	#randomize positions of b
	# random_idx = torch.randperm(half_batch)
	phrase_b = torch.roll(phrase_b, 1, dims=0)
	phrase_b_mask = torch.roll(phrase_b_mask, 1, dims=0)
	# phrase_b = phrase_b[random_idx]
	# phrase_b_mask = phrase_b_mask[random_idx]

	#add to all phrases_a the [SEP] token
	phrase_a = torch.cat((phrase_a, torch.tensor([4]).repeat(half_batch, 1)), dim=1)


	randomized_tokens = torch.cat((phrase_a, phrase_b), dim=1)
	tokens = torch.cat((randomized_tokens, tokens[half_batch:]), dim=0)

	randomized_masks = torch.cat((phrase_a_mask, phrase_b_mask), dim=1)
	masks = torch.cat((randomized_masks, masks[half_batch:]), dim=0)

	#add to all phrases in second half_batch the [SEP] token in a random position
	counts = min([torch.count_nonzero(masks[i+half_batch]) for i in range(half_batch)])//2
	sep_positions = torch.randint(1, counts, (half_batch,))
	
	tokens[half_batch:, sep_positions ] = 4
	
	tokens[:half_batch] = randomized_tokens
	masks[:half_batch] = randomized_masks

	return tokens, masks

class BertTrainer:
	
	def __init__(self, model, device, lr, model_config):
		
		self.model = model
		self.optimizer = AdamW(lr=lr, params=self.model.parameters())
		self.device = device 
		self.lm_criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
		self.cls_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6])).to(device)
		self.model_config = model_config

	def train(self, train_dataloader, eval_dataloader, num_epochs, log_freq: int, color = FOREGROUND_COLORS['Green']):
		log_step = len(train_dataloader) // log_freq
		
		scaler = GradScaler()
		scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=len(train_dataloader), num_training_steps=num_epochs*len(train_dataloader))

		self.model.train()
		train_loss = 0.0

		for epoch in range(num_epochs):

			tqdm_train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
			for step_num, batch_data in enumerate(tqdm_train_loader):
				
				self.model.zero_grad()

				tokens = batch_data["tokens"].to(self.device) 
				masks  = batch_data["attention_mask"].to(self.device)
				
				masked_tokens, cls_labels = mask_tokens(tokens, self.model_config.vocab_size, self.device)
				# masked_tokens, masks = split_and_randomize_phrases(masked_tokens, masks) #TEMPORARY REMOVED BECAUSE TOO COMPLICATED

				with autocast():
					#Model outputs (batch_size, seq_len, dict_size) and (batch_size, seq_len, 2)
					logits, label_guess = self.model(masked_tokens, masks) 
			
					lm_loss = self.lm_criterion(logits.transpose(1,2), tokens)
					cls_loss = self.cls_criterion(label_guess.squeeze(-1), cls_labels.float())

					batch_loss = lm_loss + cls_loss

			
				if step_num == 0:
					train_loss = batch_loss.item()
				else:
					train_loss = 0.9 * train_loss + 0.1 * batch_loss.item()

				tqdm_train_loader.set_postfix(loss = "{:.4f}".format(train_loss))	
				
				
				# Scales loss.  Calls backward() on scaled loss to create scaled gradients.
				# Backward passes under autocast are not recommended.
				# Backward ops run in the same dtype autocast chose for corresponding forward ops.
				scaler.scale(batch_loss).backward()

				# scaler.step() first unscales the gradients of the optimizer's assigned params.
				# If these gradients do not contain infs or NaNs, optimizer.step() is then called,
				# otherwise, optimizer.step() is skipped.
				scaler.step(self.optimizer)
				
				scaler.update()
				# Update learning rate
				scheduler.step()


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
						
						masked_tokens, cls_labels = mask_tokens(tokens, self.model_config.vocab_size, self.device)

						logits, label_guess = self.model(masked_tokens, masks)
						label_guess = label_guess.squeeze(-1)

						lm_loss = self.lm_criterion(logits.transpose(1,2), tokens)
						cls_loss = nn.functional.binary_cross_entropy_with_logits(label_guess, cls_labels.float(), pos_weight=torch.tensor([15]).to(self.device))

						val_loss += lm_loss.item() + cls_loss.item()
						cls_avg_loss += cls_loss.item()
						mlm_avg_loss += lm_loss.item()

						mlm_accuracy = torch.cat((mlm_accuracy, (torch.argmax(logits, dim=2) == tokens)))
						cls_accuracy = torch.cat((cls_accuracy, ((label_guess > 0.5) == cls_labels.float())))
					

					mlm_accuracy = torch.mean(mlm_accuracy.float()).item()					
					cls_accuracy = torch.mean(cls_accuracy.float()).item()
					val_loss = val_loss / len(eval_dataloader)
					cls_avg_loss = cls_avg_loss / len(eval_dataloader)
					mlm_avg_loss = mlm_avg_loss / len(eval_dataloader)

					tqdm.write(f"{RESET}Val loss: {val_loss:.3f}, mlm loss: {mlm_avg_loss:.3f}, mlm accuracy: {mlm_accuracy:.3f}, cls loss: {cls_avg_loss:.3f}, cls accuracy: {cls_accuracy:.3f}, Lr: {scheduler.get_last_lr()} {color}")
					self.model.train()


class Trainer:
	def __init__(self, model, device, lr, model_config):
		self.model = model
		self.optimizer = AdamW(lr=lr, params=self.model.parameters())
		self.device = device 
		self.criterion = nn.CrossEntropyLoss().to(device)
		self.model_config = model_config

	def train(self, train_dataloader, eval_dataloader, num_epochs, log_freq: int, color = FOREGROUND_COLORS['Green']):

		log_step = len(train_dataloader) // log_freq
		
		scaler = GradScaler()

		# scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=len(train_dataloader), num_training_steps=num_epochs*len(train_dataloader))
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=log_freq)
	

		self.model.train()
		train_loss = 0.0

		min_loss = float('inf')


		for epoch in range(num_epochs):

			tqdm_train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
			for step_num, batch_data in enumerate(tqdm_train_loader):
				
				self.model.zero_grad()

				tokens = batch_data["tokens"].to(self.device) 
				masks  = batch_data["attention_mask"].to(self.device)
				labels = batch_data["label"].to(self.device)
				
				with autocast():
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
				scaler.scale(batch_loss).backward()

				# scaler.step() first unscales the gradients of the optimizer's assigned params.
				# If these gradients do not contain infs or NaNs, optimizer.step() is then called,
				# otherwise, optimizer.step() is skipped.
				scaler.step(self.optimizer)
				
				scaler.update()
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
					
					scheduler.step(-mcc) 
					if(val_loss < min_loss):
						min_loss = val_loss
						checkpoint(self.model, "best_model.pth")

					tqdm.write(f"{RESET}Val loss: {val_loss:.3f}, Val accuracy: {val_accuracy:.3f}, Val mcc: {mcc:.3f}, Lr: {scheduler.get_last_lr()} {color}")
					self.model.train()
		resume(self.model, "best_model.pth")

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