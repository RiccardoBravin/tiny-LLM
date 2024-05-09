
import torch
import torch.nn as nn
from tqdm import tqdm
import random

from sklearn.metrics import f1_score, matthews_corrcoef

class Generator(nn.Module):
	def __init__(self, model, d_model, vocab_size):
		super().__init__()
		self.model = model
		self.norm = nn.LayerNorm(d_model)
		self.lm = nn.Linear(d_model, vocab_size)
	
	def forward(self, x):
		x = self.model(x)
		x = self.norm(x)
		x = self.lm(x)
		return x
	

class Electra(nn.Module):
	def __init__(self, gen_model, disc_model, d_model, vocab_size):
		super().__init__()
		self.gen = Generator(gen_model, d_model, vocab_size)
		self.discriminator = disc_model
		self.norm = nn.LayerNorm(d_model)
		self.lm = nn.Linear(d_model, 2)
	
	def forward(self, x):
		
		mask = torch.rand(x.size()) > 0.15
		#print("mask ", mask)
		masked_in = mask * x
		generated = self.gen(masked_in)
		generated_token = torch.argmax(generated, dim=-1)
		#print("generated_token ", generated_token)

		different = (generated_token != x) * mask


		disc_in = x * different + generated_token * (~different) 
		#print(disc_in)
		

		x = self.discriminator(disc_in)
		x = self.norm(x)
		x = self.lm(x)

		return x, generated, different
	

class SimpleElectra (nn.Module):
	def __init__(self, disc_model, d_model, vocab_size):
		super().__init__()
		self.vocab_size = vocab_size
		
		self.discriminator = disc_model
		self.norm = nn.LayerNorm(d_model)
		self.fc = nn.Linear(d_model, 2)
	def forward(self, x):
		#print("x ", x[0])
		ranges = torch.count_nonzero(x, dim=1)
		generated_token = x
		disc_label = torch.zeros(x.size()).to(x.device)
		
		for i in range(x.size(0)):
			for j in range(ranges[i]):
				if random.random() < 0.10:
					rand_tok = random.randint(3, self.vocab_size-1)
					if rand_tok != generated_token[i][j]:
						generated_token[i][j] = rand_tok
						disc_label[i][j] = 1

		#print("generated_token ", generated_token[0])
		# mask = torch.rand(x.size()) < 0.15
		# mask = mask.int().to(x.device)
		# #print("mask ",mask)
		# rand_tok = torch.randint(0, self.vocab_size, x.size()).to(x.device)
		# #print("rand_tok ", rand_tok)
		# generated_token = x * (1 - mask) + rand_tok * mask
		# #print("generated_token ", generated_token)
		# disc_label = (generated_token != x) * mask
		# #print("different", different)
		#print(torch.all(different == mask))

		x = self.discriminator(generated_token)
		x = self.norm(x)
		x = self.fc(x)

		return x, disc_label


def electraTrainer(model, train_dataloader, val_dataloader, lr, epochs):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	criterion = torch.nn.CrossEntropyLoss()
	criterion.to(device);

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*2, eta_min=1e-6)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
	log_step = len(train_dataloader) // 10

	for epoch in range(epochs):
		tqdm.write(f"Epoch {epoch+1}/{epochs}")
		
		model.train()
		train_loss = 0
		tqdm_train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

		for step_num, batch_data in enumerate(tqdm_train_loader):

			token_ids, labels = batch_data
			token_ids, labels = token_ids.to(device), labels.to(device)

			logits, made_labels = model(token_ids)

			mask = token_ids > 0
			logits = logits.view(-1, 2)  # Ridimensiona a (batch_size * len, 2)
			made_labels = made_labels.view(-1).long()  # Ridimensiona a (batch_size * len)
			mask = mask.view(-1)

			logits = logits[mask]
			made_labels = made_labels[mask]

			batch_loss = criterion(logits, made_labels)
				  
			if step_num == 0:
				train_loss = batch_loss.item()
			else:
				train_loss = 0.9 * train_loss + 0.1 * batch_loss.item()

			model.zero_grad()
			batch_loss.backward()


			#torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
			

			loss_str = "{:.4f}".format(train_loss)
			tqdm_train_loader.set_postfix(loss = loss_str)			
			if step_num % log_step == (log_step - 1):
				model.eval()
				val_accuracy = []
				val_loss = 0
				for x, y in val_dataloader:
					x, y = x.to(device), y.to(device)

					guess, lab = model(x)
					
					mask = x > 0
					mask = mask.view(-1)

					guess = guess.view(-1, 2)  # Ridimensiona a (batch_size * len, 2)
					lab = lab.view(-1).long()  # Ridimensiona a (batch_size * len)

					guess = guess[mask]
					lab = lab[mask]

					val_loss += criterion(guess, lab).item()
					val_accuracy += (torch.argmax(guess, dim=1) == lab).tolist()
				
				val_accuracy = sum(val_accuracy) / len(val_accuracy)
				mcc = matthews_corrcoef(lab.cpu().numpy(), torch.argmax(guess, dim=1).cpu().numpy())
				val_loss = val_loss / len(val_dataloader)
				scheduler.step(-mcc)
				tqdm.write(f"Val loss: {val_loss:.3f}, Val accuracy: {val_accuracy:.3f}, Val mcc: {mcc:.3f}, Lr: {scheduler.get_last_lr()[0]:.6f}")
				model.train()
	
def electraEvaluator(model, test_dataloader):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	criterion = torch.nn.CrossEntropyLoss()
	criterion.to(device);

	model.eval()
	predicted = torch.Tensor([])
	all_logits = torch.Tensor([])
	all_labels = torch.Tensor([]).to(device)

	tqdm_test_loader = tqdm(test_dataloader, desc=f"Evaluation: ", leave=False)
	
	avg_loss = 0
	with torch.no_grad():
		for step_num, batch_data in enumerate(tqdm_test_loader):

			token_ids, labels = batch_data
			token_ids, labels = token_ids.to(device), labels.to(device)

			logits, made_labels = model(token_ids)
			
			logits = logits.view(-1, 2)  # Ridimensiona a (batch_size * len, 2)
			made_labels = made_labels.view(-1).long()  # Ridimensiona a (batch_size * len)

			loss = criterion(logits, made_labels)
			avg_loss += loss.item()
			numpy_logits = logits.cpu().detach()
			predicted = torch.cat((predicted, torch.argmax(numpy_logits, dim = 1)))
			#all_logits  = torch.cat((all_logits, numpy_logits))
			all_labels = torch.cat((all_labels, made_labels))
			
	
	average_loss = avg_loss / len(test_dataloader)
	print("average loss: ", average_loss)
	print()
	
	return predicted.tolist(), average_loss, all_labels;